"""Audit Q1 + Q2 analysis: exact MAP and corrected beam search vs the cheap rules, per AR format.

python ablation/beamfix_analyze.py [--v2 outputs/ablation_v2] [--src outputs/ablation]

Inputs (pulled from the VMs):
  <v2>/raw/beam_fix/<run>/beam_rows.json + beam.json   corrected beam (beam_ab --keep_none), b{4,8,16} x lp{0,1}
  <v2>/raw/exactmap/<run>/map_rows.json + map.json     exact MAP (exactmap_ab)
  <src>/<run>/rows_c.json                              seed-0 constrained eval rows (threshold, greedy, proxy)
Each file may be an image subsample (image_stride); reference rules are recomputed on the same images.
Rules (present refusal on RefCOCO val single-target / N-acc on gRefCOCO no-target):
  thresh       P(none) >= .5            greedy_c   grammar-constrained greedy first token
  proxy        greedy-candidate MAP: log q(none <end>) > log q(greedy box)  (upper bound on exact-MAP refusal)
  beam*_fixed  corrected beam (none always a finished hypothesis), lp0 = sum log-prob, lp1 = / length
  beam*_hf     what HF beam search would do: same, but "none" only survives if it ranks in the top 2B first tokens
               (none_rank_step1 < 2B); otherwise the best box is returned
  exact_map    exact MAP (best-first search pruned by q(none)); "undecided" rows reported separately
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = ["ar_digits_neg25", "ar_hier_neg25", "ar_atomic100_neg25", "ar_atomic1000_neg25"]
LABEL = {"ar_digits_neg25": "AR-digits", "ar_hier_neg25": "AR-hier", "ar_atomic100_neg25": "AR-atomic-100",
         "ar_atomic1000_neg25": "AR-atomic-1000"}


def groups_of(prep):
    k, groups = 0, []
    for l in open(prep):
        r = json.loads(l)
        g = []
        for q in r["questions"]:
            if not (q["present"] and q.get("n_targets", 1) > 1):
                g.append(k)
            k += 1
        if g:
            groups.append(g)
    return groups


def ci(x, B=1000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="outputs/ablation_v2")
    ap.add_argument("--src", default="outputs/ablation")
    ap.add_argument("--prep", default="data/prep/eval_grefcoco_val_merged.jsonl")
    a = ap.parse_args()
    G = groups_of(a.prep)
    res = {}
    for run in RUNS:
        ref = json.load(open(f"{a.src}/{run}/rows_c.json"))["grefcoco_val_merged"]
        out = dict(rules={})
        srcs = {}
        p = f"{a.v2}/raw/beam_fix/{run}/beam_rows.json"
        if os.path.exists(p):
            srcs["beam"] = (json.load(open(p)), json.load(open(p.replace("beam_rows", "beam"))).get("image_stride", 1))
        p = f"{a.v2}/raw/exactmap/{run}/map_rows.json"
        if os.path.exists(p):
            srcs["map"] = (json.load(open(p)), json.load(open(p.replace("map_rows", "map"))).get("image_stride", 1))
        if not srcs:
            continue
        for kind, (rows, stride) in srcs.items():
            sel = [i for g in G[::stride] for i in g][: len(rows)]
            R = [ref[i] for i in sel]
            assert len(R) == len(rows) and all(r["present"] == b["present"] for r, b in zip(R, rows)), (run, kind)
            pres = np.array([r["present"] for r in R]); absn = ~pres
            tag = f"stride{stride}"
            base = {"thresh": np.array([r["p_none"] >= .5 for r in R]),
                    "greedy_c": np.array([r["greedy_none_c"] for r in R]),
                    "proxy (greedy-cand. MAP)": np.array([r["seqmap_none"] for r in R])}
            rules = dict(base)
            if kind == "beam":
                confs = [c for c in rows[0] if c[0] == "b" and "lp" in c]
                rk = np.array([b["none_rank_step1"] for b in rows])
                out["none_outside_top2B_step1_present"] = {f"B={B}": float((rk[pres] >= 2 * B).mean()) for B in (4, 8, 16)}
                for c in confs:
                    B = int(c[1:].split("lp")[0])
                    fx = np.array([b[c]["kind"] == "none" for b in rows])
                    rules[f"{c}_fixed"] = fx
                    rules[f"{c}_hf"] = fx & (rk < 2 * B)
                for c in confs:
                    iou = np.array([b[c]["iou"] if b[c]["iou"] is not None else 0.0 for b in rows])
                    acc = pres & ~rules[f"{c}_fixed"]
                    out.setdefault("acc50_accepted", {})[c] = float((iou[acc] >= .5).mean())
            else:
                st = np.array([b["status"] for b in rows])
                rules["exact_map"] = st == "none"
                out["exact_map_undecided"] = dict(present=float((st[pres] == "undecided").mean()),
                                                  absent=float((st[absn] == "undecided").mean()))
                out["exact_map_refuse_upper"] = float(((st == "none") | (st == "undecided"))[pres].mean())
                # does the proxy agree with exact MAP? (proxy refuses but exact MAP accepts = proxy over-refusal)
                prx = base["proxy (greedy-cand. MAP)"]
                dec = st != "undecided"
                out["proxy_vs_exact_present"] = dict(
                    both_refuse=int((pres & dec & prx & (st == "none")).sum()),
                    proxy_refuse_exact_accept=int((pres & dec & prx & (st == "box")).sum()),
                    proxy_accept_exact_refuse=int((pres & dec & ~prx & (st == "none")).sum()),
                    n_present=int(pres.sum()))
                iou = np.array([r.get("iou", 0.0) for r in R])
                ref_ = pres & (st == "none"); acc_ = pres & (st == "box")
                out["exact_map_refused_present_greedy_box"] = dict(acc50=float((iou[ref_] >= .5).mean()) if ref_.any() else None,
                                                                   mean_iou=float(iou[ref_].mean()) if ref_.any() else None)
                out["exact_map_accepted_present_greedy_box_acc50"] = float((iou[acc_] >= .5).mean())
            for name, rf in rules.items():
                key = name if name in base and kind == "beam" else name
                if name in base:
                    key = f"{name} [{kind} subset]"
                out["rules"][key] = dict(refusal_present=float(rf[pres].mean()), refusal_present_ci95=ci(rf[pres]),
                                         N_acc=float(rf[absn].mean()), n_present=int(pres.sum()), n_absent=int(absn.sum()),
                                         subset=f"{kind} {tag}")
        res[run] = out
        print(run, json.dumps({k: (round(v["refusal_present"] * 100, 1), round(v["N_acc"] * 100, 1)) for k, v in out["rules"].items()}), flush=True)
    os.makedirs(f"{a.v2}/beam_fix", exist_ok=True)
    json.dump(res, open(f"{a.v2}/beam_fix/beamfix_summary.json", "w"), indent=1)
    # plot
    runs = [r for r in RUNS if r in res]
    show = [("thresh [beam subset]", "P(none)≥.5", "#9aa0a6"), ("greedy_c [beam subset]", "greedy", "#5f6368"),
            ("b4lp0_hf", "HF beam-4 (none pruned)", "#a8c7fa"), ("b4lp0_fixed", "beam-4 lp0 (fixed)", "#f6aea9"),
            ("b8lp0_fixed", "beam-8 lp0", "#ef7d73"), ("b16lp0_fixed", "beam-16 lp0", "#ea4335"),
            ("b16lp1_fixed", "beam-16 lp1 (len-norm)", "#4285f4"), ("exact_map", "exact MAP", "#7b1fa2"),
            ("proxy (greedy-cand. MAP) [beam subset]", "greedy-cand. MAP proxy", "#000000")]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.8 / len(show)
    for j, (k, lab, col) in enumerate(show):
        ys = [res[r]["rules"].get(k, {}).get("refusal_present", np.nan) * 100 for r in runs]
        xs = np.arange(len(runs)) + (j - len(show) / 2 + .5) * w
        ax.bar(xs, ys, w * .95, color=col, label=lab)
        for x, y in zip(xs, ys):
            if np.isfinite(y):
                ax.text(x, y + .5, f"{y:.1f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(range(len(runs))); ax.set_xticklabels([LABEL[r] for r in runs])
    ax.set_ylabel("present targets refused (%)"); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7, ncol=3)
    ax.set_title("Same trained models, different decoders: refusal on RefCOCO val present targets", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.v2}/beam_fix/beamfix_refusal.png", dpi=150)


if __name__ == "__main__":
    main()
