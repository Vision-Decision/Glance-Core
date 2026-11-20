"""Experiment 1: multi-seed summary (gate vs AR-digits at 25% and 5% training negatives).

python ablation/seeds_v2.py [--out outputs/ablation_v2/seeds]
Seed 0 = outputs/ablation/<run>/, seeds 1,2 = outputs/ablation_v2/runs/<run>_s<k>/.  AR rows = constrained decoding.

Per run (= head x neg ratio x seed) metrics:
  acc50_{val,testA,testB}            RefCOCO Acc@0.5 (AR: grammar-constrained box)
  auroc, ece                         presence score (gate p_present; AR 1-P(none)) on gRefCOCO-val-merged
  N_acc@1%                           N-acc when the threshold refuses exactly 1% of RefCOCO-val present targets
  N_acc_<rule>, halluc_<rule>        gRefCOCO no-target (8,905)
  refusal_<rule>                     RefCOCO val present (10,834)
  rules: gate (p>=.5) | greedy (unconstrained first token argmax) | greedy_c (argmax over grammar-valid tokens)
         | thresh (P(none)<.5) | seqmap (seq-MAP proxy, greedy box as the competitor)
Uncertainty: mean +- sample std over seeds; 95% CI of the seed-mean from an image-level cluster bootstrap
(1000 resamples, the same resampled images for every seed and head, so differences are paired).
"""
import argparse, json, os
import numpy as np

SRC0, SRC1 = "outputs/ablation", "outputs/ablation_v2/runs"
CELLS = [("ours", "neg25"), ("ar_digits", "neg25"), ("ours", "neg05"), ("ar_digits", "neg05")]


def rows_of(run, seed):
    d = f"{SRC0}/{run}" if seed == 0 else f"{SRC1}/{run}_s{seed}"
    for f in ("rows_c.json", "rows.json"):
        if os.path.exists(f"{d}/{f}"):
            return json.load(open(f"{d}/{f}"))
    return None


def group_ids(path, rows_len):
    g = []
    for i, l in enumerate(open(path)):
        g += [i] * len(json.loads(l)["questions"])
    assert len(g) == rows_len, (len(g), rows_len)
    return np.array(g)


def auroc(p, y):
    from scipy.stats import rankdata
    r = rankdata(p)
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ece(p, y, n_bins=15):
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


class Run:
    """Pre-extracted arrays for one run."""

    def __init__(self, rows, head):
        G = rows["grefcoco_val_merged"]
        self.head = head
        self.pres = np.array([r["present"] and "iou" in r for r in G])
        self.absn = np.array([not r["present"] for r in G])
        self.iou = np.array([r.get("iou", 0.0) for r in G])
        if head == "ours":
            self.score = np.array([r["p_present"] for r in G])
            self.say = {"gate": self.score >= .5}
        else:
            pn = np.array([r["p_none"] for r in G])
            self.score = 1 - pn
            self.say = {"greedy": ~np.array([r["greedy_none"] for r in G]),
                        "greedy_c": ~np.array([r["greedy_none_c"] for r in G]),
                        "thresh": pn < .5, "seqmap": ~np.array([r["seqmap_none"] for r in G])}
        self.test = {s: np.array([r["iou"] for r in rows[s] if r["present"] and "iou" in r]) for s in ("refcoco_testA", "refcoco_testB")}

    def metrics(self, w=None):
        """w: per-row multiplicity weights (bootstrap); None = original sample."""
        if w is None:
            w = np.ones(len(self.pres))
        out = {}
        wp, wa = w * self.pres, w * self.absn
        out["acc50_val"] = float((wp * (self.iou >= .5)).sum() / wp.sum())
        out["mean_iou_val"] = float((wp * self.iou).sum() / wp.sum())
        for rule, s in self.say.items():
            out[f"N_acc_{rule}"] = float((wa * ~s).sum() / wa.sum())
            out[f"halluc_{rule}"] = 1 - out[f"N_acc_{rule}"]
            out[f"refusal_{rule}"] = float((wp * ~s).sum() / wp.sum())
        # weighted AUROC / ECE / matched refusal via repetition
        idx = np.repeat(np.arange(len(w)), w.astype(int))
        sc, pr, ab = self.score[idx], self.pres[idx], self.absn[idx]
        m = pr | ab
        out["auroc"] = auroc(sc[m], pr[m].astype(int))
        out["ece"] = ece(sc[m], pr[m].astype(float))
        thr = np.quantile(sc[pr], 0.01)
        out["N_acc@1%"] = float((sc[ab] < thr).mean())
        return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ablation_v2/seeds")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = {}
    for head, neg in CELLS:
        for seed in (0, 1, 2):
            rows = rows_of(f"{head}_{neg}", seed)
            if rows is not None:
                runs[(head, neg, seed)] = (Run(rows, head), rows)
    n = len(next(iter(runs.values()))[1]["grefcoco_val_merged"])
    gid = group_ids("data/prep/eval_grefcoco_val_merged.jsonl", n)
    n_img = gid.max() + 1
    per_run = {f"{h}_{g}_s{s}": dict(r.metrics(), acc50_testA=float((r.test["refcoco_testA"] >= .5).mean()),
                                     acc50_testB=float((r.test["refcoco_testB"] >= .5).mean()))
               for (h, g, s), (r, _) in runs.items()}
    rng = np.random.default_rng(0)
    boots = {k: [] for k in runs}
    for b in range(a.boot):
        imgs = rng.integers(0, n_img, n_img)
        cnt = np.bincount(imgs, minlength=n_img)
        w = cnt[gid].astype(float)
        for k, (r, _) in runs.items():
            boots[k].append(r.metrics(w))
    summary = {}
    for head, neg in CELLS:
        seeds = sorted(s for (h, g, s) in runs if h == head and g == neg)
        if not seeds:
            continue
        keys = per_run[f"{head}_{neg}_s{seeds[0]}"].keys()
        cell = dict(seeds=seeds)
        for k in keys:
            vals = [per_run[f"{head}_{neg}_s{s}"][k] for s in seeds]
            bm = [np.mean([boots[(head, neg, s)][b][k] for s in seeds]) for b in range(a.boot)] if k in boots[(head, neg, seeds[0])][0] else None
            cell[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                           per_seed=vals,
                           ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))] if bm else None)
        summary[f"{head}_{neg}"] = cell
    # paired differences (AR-digits - gate) for shared metrics, seed-mean, image bootstrap
    diffs = {}
    for neg in ("neg25", "neg05"):
        sg = sorted(s for (h, g, s) in runs if h == "ours" and g == neg)
        sa = sorted(s for (h, g, s) in runs if h == "ar_digits" and g == neg)
        common = sorted(set(sg) & set(sa))
        if not common:
            continue
        d = {}
        for k, rule_a, rule_g in [("N_acc@1%", None, None), ("auroc", None, None), ("ece", None, None),
                                  ("acc50_val", None, None), ("N_acc", "thresh", "gate"), ("N_acc", "greedy", "gate"),
                                  ("refusal", "thresh", "gate"), ("refusal", "greedy", "gate"), ("refusal", "seqmap", "gate")]:
            ka = f"{k}_{rule_a}" if rule_a else k
            kg = f"{k}_{rule_g}" if rule_g else k
            point = np.mean([per_run[f"ar_digits_{neg}_s{s}"][ka] - per_run[f"ours_{neg}_s{s}"][kg] for s in common])
            bs = [np.mean([boots[("ar_digits", neg, s)][b][ka] - boots[("ours", neg, s)][b][kg] for s in common]) for b in range(a.boot)]
            d[f"{ka} - gate:{kg}"] = dict(diff=float(point), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                          seeds=common)
        diffs[neg] = d
    # 25% -> 5% hallucination increase per head (seed-paired where possible)
    shift = {}
    for head, rule in (("ours", "gate"), ("ar_digits", "thresh"), ("ar_digits", "greedy")):
        common = sorted(set(s for (h, g, s) in runs if h == head and g == "neg25") & set(s for (h, g, s) in runs if h == head and g == "neg05"))
        if common:
            k = f"halluc_{rule}"
            point = np.mean([per_run[f"{head}_neg05_s{s}"][k] - per_run[f"{head}_neg25_s{s}"][k] for s in common])
            bs = [np.mean([boots[(head, "neg05", s)][b][k] - boots[(head, "neg25", s)][b][k] for s in common]) for b in range(a.boot)]
            shift[f"{head}/{rule}"] = dict(diff=float(point), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], seeds=common)
    out = dict(per_run=per_run, summary=summary, diff_ar_minus_gate=diffs, halluc_increase_25_to_5=shift,
               n_images=int(n_img), boot=a.boot)
    json.dump(out, open(f"{a.out}/seeds.json", "w"), indent=1)
    print(json.dumps(diffs, indent=1)); print(json.dumps(shift, indent=1))
    plot(summary, per_run, a.out)


def plot(summary, per_run, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mets = [("acc50_val", "RefCOCO val Acc@0.5"), ("N_acc@1%", "N-acc @1% matched refusal"), ("auroc", "presence AUROC"),
            ("halluc_main", "hallucination (default rule)"), ("refusal_seqmap", "present refusal, seq-MAP")]
    fig, axes = plt.subplots(1, len(mets), figsize=(3.2 * len(mets), 3.4))
    cells = [c for c in ("ours_neg25", "ar_digits_neg25", "ours_neg05", "ar_digits_neg05") if c in summary]
    col = {"ours": "#2a78d6", "ar_digits": "#eb6834"}
    for ax, (k, title) in zip(axes, mets):
        for i, c in enumerate(cells):
            head = "ar_digits" if c.startswith("ar") else "ours"
            kk = k if k != "halluc_main" else ("halluc_gate" if head == "ours" else "halluc_greedy")
            if kk not in summary[c]:
                continue
            v = np.array(summary[c][kk]["per_seed"]) * 100
            ax.scatter([i] * len(v), v, color=col[head], s=18, alpha=.6)
            m = summary[c][kk]["mean"] * 100
            ci = summary[c][kk]["ci95"]
            ax.errorbar([i + .18], [m], yerr=[[m - ci[0] * 100], [ci[1] * 100 - m]] if ci else None, fmt="o",
                        color=col[head], ms=6, capsize=3)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([c.replace("ar_digits", "AR-dig").replace("ours", "gate").replace("_neg", "\n") + "%" for c in cells], fontsize=8)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(alpha=.3)
    fig.suptitle("Multi-seed (dots = seeds; marker+bar = seed mean, 95% image-bootstrap CI)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/seeds.png", dpi=150)


if __name__ == "__main__":
    main()
