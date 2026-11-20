"""Experiments 2 + 3: beam search vs the seq-MAP proxy, and the fixed-model refinement test.

python ablation/beam_v2_analyze.py [--beam_dir outputs/ablation_v2/beam] [--out outputs/ablation_v2/beam]

Inputs: <beam_dir>/<run>/beam_rows.json (ablation/beam_ab.py) aligned with outputs/ablation/<run>/rows_c.json
(seed-0 constrained eval rows, multi-target rows removed). Present = RefCOCO val (single-target),
absent = gRefCOCO val no-target. If beam_ab ran on an image subsample (--image_stride), the rows_c reference
numbers are recomputed on the same subsample.

Rules compared (present refusal / N-acc):
  thresh        P(none) >= 0.5 (consistent rule)
  greedy_c      grammar-constrained greedy (first token argmax over valid tokens)
  seqmap_proxy  log q(none <end>) > log q(greedy box)  (the earlier proxy, an upper bound on exact-MAP refusal)
  b{B}lp{L}     beam search; refuse iff the best finished hypothesis is "none <end>"
  map_est       log q(none <end>) > max over every box hypothesis we found (greedy box, all beam n-best lists, lp0
                and lp1). Upper bound on exact-MAP refusal, tighter than the proxy.
Refinement (Prop. 2 iii): refine each coordinate token into k equivalent surface forms with q/k each.
  split{k}_b8lp0      beam search run on the refined model (actual decoding)
  map_est_split{k}    analytic exact-MAP estimate on the refined space: refuse iff log q(none) > max_s [log q(s) -
                      n_coord(s) log k] over all found box hypotheses (n_coord = 4 atomic, 8 hier, #digits for digits)
  greedy_split{k}     analytic: refuse iff q(none) > max valid first coordinate token prob / k
  thresh              unchanged by construction (P(none) is not split)
Digits, leading zeros: each number v (written with d digits) also has the forms with 1..4-d leading zeros
(width <= 4), r'(s) = r(s)/prod_i (5 - d_i). map_est_leadzero = log q(none) > max_s [log q(s) - sum log(5-d_i)].
"""
import argparse, json, math, os
import numpy as np

RUNS = ["ar_digits_neg25", "ar_hier_neg25", "ar_atomic100_neg25", "ar_atomic1000_neg25"]
LABEL = {"ar_digits_neg25": "AR-digits", "ar_hier_neg25": "AR-hier", "ar_atomic100_neg25": "AR-atomic-100",
         "ar_atomic1000_neg25": "AR-atomic-1000"}


def ndig(box):
    return [len(str(int(round(v * 1000)))) for v in box]


def n_coord(run, box):
    if "digits" in run:
        return sum(ndig(box))
    return 8 if "hier" in run else 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam_dir", default="outputs/ablation_v2/beam")
    ap.add_argument("--src", default="outputs/ablation")
    ap.add_argument("--out", default="outputs/ablation_v2/beam")
    a = ap.parse_args()
    res = {}
    for run in RUNS:
        p = f"{a.beam_dir}/{run}/beam_rows.json"
        if not os.path.exists(p):
            continue
        B = json.load(open(p))
        summ = json.load(open(f"{a.beam_dir}/{run}/beam.json"))
        stride = summ.get("image_stride", 1)
        ref = json.load(open(f"{a.src}/{run}/rows_c.json"))["grefcoco_val_merged"]
        # rebuild per-record grouping to apply the same image subsample
        recs = [json.loads(l) for l in open("data/prep/eval_grefcoco_val_merged.jsonl")]
        k = 0; groups = []
        for r in recs:
            g = []
            for q in r["questions"]:
                if not (q["present"] and q.get("n_targets", 1) > 1):
                    g.append(k)
                k += 1
            if g:
                groups.append(g)
        sel = [i for g in groups[::stride] for i in g][: len(B)]
        R = [ref[i] for i in sel]
        assert len(R) == len(B), (len(R), len(B))
        dp = np.abs(np.array([r["p_none"] for r in R]) - np.array([b["p_none"] for b in B]))
        pres = np.array([r["present"] for r in R]); absn = ~pres
        lpn = np.array([r["lp_none"] for r in R])
        pn = np.array([r["p_none"] for r in R])
        confs = [c for c in B[0] if c.startswith("b") or c.startswith("split")]
        refuse = {"thresh": pn >= .5, "greedy_c": np.array([r["greedy_none_c"] for r in R]),
                  "seqmap_proxy": np.array([r["seqmap_none"] for r in R])}
        for c in confs:
            refuse[c] = np.array([b[c]["kind"] == "none" for b in B])
        # every box hypothesis found (unrefined configs only): (log q(s), box)
        cands = []
        for r, b in zip(R, B):
            cs = [(r["lp_box"], r["box"])] if r.get("box") else []
            for c in confs:
                if c.startswith("split"):
                    continue
                cs += [(h["lp"], h["box"]) for h in b[c]["nbest"] if h["kind"] == "box"]
            cands.append(cs)
        Mhat = np.array([max(x[0] for x in cs) for cs in cands])
        refuse["map_est"] = lpn > Mhat
        out = dict(n_present=int(pres.sum()), n_absent=int(absn.sum()), image_stride=stride,
                   p_none_max_abs_diff_vs_eval=float(dp.max()), timing_s=summ.get("timing_s"), rules={})
        # refinement analytics
        pmc = np.array([r["p_maxcoord_c"] for r in R])
        for kk in (2, 4, 10):
            Mk = np.array([max(lp - n_coord(run, bx) * math.log(kk) for lp, bx in cs) for cs in cands])
            refuse[f"map_est_split{kk}"] = lpn > Mk
            refuse[f"greedy_split{kk}"] = pn > pmc / kk
        if "digits" in run:
            Mlz = np.array([max(lp - sum(math.log(5 - d) for d in ndig(bx)) for lp, bx in cs) for cs in cands])
            refuse["map_est_leadzero"] = lpn > Mlz
            out["leadzero_mean_log_nforms_present"] = float(np.mean([sum(math.log(5 - d) for d in ndig(r["box"])) for r in R if r["present"] and r.get("box")]))
        iou_ref = np.array([r.get("iou", 0.0) for r in R])
        for name, rf in refuse.items():
            d = dict(refusal_present=float(rf[pres].mean()), N_acc=float(rf[absn].mean()))
            if name in confs:
                iou_b = np.array([b[name]["iou"] if b[name]["iou"] is not None else 0.0 for b in B])
                acc = pres & ~rf
                d["acc50_of_accepted_present"] = float((iou_b[acc] >= .5).mean())
                d["acc50_refused_present_greedybox"] = float((iou_ref[pres & rf] >= .5).mean()) if (pres & rf).any() else None
                d["bad"] = float(np.mean([b[name]["kind"] == "bad" for b in B]))
            res_n = int((pres & rf).sum())
            d["n_refused_present"] = res_n
            out["rules"][name] = d
        # nesting check under beam (lp0): refusal(thresh) <= greedy <= beam? report fractions violating
        for c in [c for c in confs if c.endswith("lp0") and c.startswith("b")]:
            out["rules"][c]["beam_refuses_but_proxy_accepts"] = int((pres & refuse[c] & ~refuse["seqmap_proxy"]).sum())
            out["rules"][c]["proxy_refuses_but_beam_accepts"] = int((pres & ~refuse[c] & refuse["seqmap_proxy"]).sum())
        if "greedy_inc" in B[0]:
            dl = [abs(b["greedy_inc"]["lp"] - r["lp_box"]) for r, b in zip(R, B)]
            same = [b["greedy_inc"]["box"] is not None and r["box"] is not None and
                    np.allclose(b["greedy_inc"]["box"], r["box"], atol=1e-4) for r, b in zip(R, B)]
            out["incremental_greedy_check"] = dict(n=len(dl), max_abs_dlp=float(max(dl)), median_abs_dlp=float(np.median(dl)),
                                                   same_box_frac=float(np.mean(same)))
        res[run] = out
        print(run, json.dumps({k: (v["refusal_present"], v["N_acc"]) for k, v in out["rules"].items()}), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/beam_summary.json", "w"), indent=1)
    plot(res, a.out)


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [r for r in RUNS if r in res]
    if not runs:
        return
    rules = ["thresh", "greedy_c", "b4lp1", "b8lp1", "b4lp0", "b8lp0", "map_est", "seqmap_proxy"]
    lab = ["P(none)≥.5", "greedy", "beam4 lp1", "beam8 lp1", "beam4 lp0", "beam8 lp0", "MAP est.", "seq-MAP proxy"]
    cols = ["#9aa0a6", "#5f6368", "#a8c7fa", "#4285f4", "#f6aea9", "#ea4335", "#7b1fa2", "#000000"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    w = 0.8 / len(rules)
    for ax, key, title in [(axes[0], "refusal_present", "present targets refused (%)"), (axes[1], "N_acc", "N-acc on no-target (%)")]:
        for j, (rn, l, c) in enumerate(zip(rules, lab, cols)):
            ys = [res[r]["rules"].get(rn, {}).get(key, np.nan) * 100 for r in runs]
            xs = np.arange(len(runs)) + (j - len(rules) / 2 + .5) * w
            ax.bar(xs, ys, w * .95, color=c, label=l)
            for x, y in zip(xs, ys):
                if np.isfinite(y) and key == "refusal_present":
                    ax.text(x, y + .5, f"{y:.1f}", ha="center", fontsize=6, rotation=90)
        ax.set_xticks(range(len(runs))); ax.set_xticklabels([LABEL[r] for r in runs])
        ax.set_title(title, loc="left"); ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{out}/beam_vs_proxy.png", dpi=150)
    # refinement plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, r in zip(axes, [x for x in ("ar_atomic100_neg25", "ar_digits_neg25") if x in res]):
        R = res[r]["rules"]
        ks = [1, 2, 4, 10]
        series = [("beam8 lp0 on refined model", [R.get("b8lp0", {}).get("refusal_present")] + [R.get(f"split{k}_b8lp0", {}).get("refusal_present") for k in ks[1:]], "#ea4335", "o-"),
                  ("exact-MAP estimate (refined)", [R["map_est"]["refusal_present"]] + [R[f"map_est_split{k}"]["refusal_present"] for k in ks[1:]], "#7b1fa2", "s--"),
                  ("greedy (refined, analytic)", [R["greedy_c"]["refusal_present"]] + [R[f"greedy_split{k}"]["refusal_present"] for k in ks[1:]], "#5f6368", "^-."),
                  ("P(none)≥.5 (invariant)", [R["thresh"]["refusal_present"]] * 4, "#9aa0a6", "d:")]
        for name, ys, c, st in series:
            xs = [k for k, y in zip(ks, ys) if y is not None]
            ax.plot(xs, [y * 100 for y in ys if y is not None], st, color=c, label=name, lw=1.8, ms=5)
        if "map_est_leadzero" in R:
            ax.axhline(R["map_est_leadzero"]["refusal_present"] * 100, color="#7b1fa2", lw=1, ls=":",
                       label=f"leading-zero forms, MAP est. ({R['map_est_leadzero']['refusal_present']*100:.1f}%)")
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k = surface forms per coordinate token (same trained model)")
        ax.set_ylabel("present targets refused (%)")
        ax.set_title(LABEL[r], loc="left"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/refinement.png", dpi=150)


if __name__ == "__main__":
    main()
