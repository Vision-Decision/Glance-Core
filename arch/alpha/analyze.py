"""Exp 1 analysis: LoRA-delta scaling sweep. python arch/alpha/analyze.py DIR(with v1.1/ v1.2/ a<alpha>/...) --out outputs/arch/alpha
SS-Pro: click accuracy per alpha with paired bootstrap CI of the difference to alpha=1 (same questions); RefCOCO testA/B
Acc@0.5; gRefCOCO val merged: single-target Acc@0.5, N-acc at p>=0.5 and N-acc at the threshold that keeps the
alpha=1 true-accept rate (presence recalibration, label-free on absent rows is impossible, so this is an oracle-free
'same recall' operating point computed on present rows), AUROC."""
import argparse, glob, gzip, json, os
import numpy as np


def auroc(s, y):
    s, y = np.asarray(s), np.asarray(y, bool)
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ci(x, n=1000):
    x = np.asarray(x, float); rng = np.random.default_rng(0)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for model in sorted(os.listdir(a.dir)):
        M = {}
        al = sorted({float(os.path.basename(d)[1:]) for d in glob.glob(f"{a.dir}/{model}/a*")})
        ss = {}
        for x in al:
            p = f"{a.dir}/{model}/a{x:g}/sspro.rows.jsonl.gz"
            if os.path.exists(p):
                R = sorted((json.loads(l) for l in gzip.open(p, "rt")), key=lambda r: str(r["id"]))
                ss[x] = np.array([r["gt"][0] <= (r["box"][0] + r["box"][2]) / 2 <= r["gt"][2] and r["gt"][1] <= (r["box"][1] + r["box"][3]) / 2 <= r["gt"][3] for r in R], float)
        ref_g = None
        for x in al:
            d = dict(alpha=x)
            if x in ss:
                d["sspro"] = float(ss[x].mean()); d["sspro_ci95"] = ci(ss[x])
                if 1.0 in ss and x != 1.0:
                    d["sspro_delta_vs_a1"] = float((ss[x] - ss[1.0]).mean()); d["sspro_delta_ci95"] = ci(ss[x] - ss[1.0])
            for s in ("refcoco_testA", "refcoco_testB"):
                p = f"{a.dir}/{model}/a{x:g}/{s}.jsonl.gz"
                if os.path.exists(p):
                    R = [json.loads(l) for l in gzip.open(p, "rt")]
                    d[s] = float(np.mean([r["iou"] >= 0.5 for r in R if "iou" in r]))
            p = f"{a.dir}/{model}/a{x:g}/grefcoco_val_merged.jsonl.gz"
            if os.path.exists(p):
                R = [json.loads(l) for l in gzip.open(p, "rt")]
                pr = np.array([r["p_present"] for r in R]); y = np.array([r["present"] for r in R])
                single = [r for r in R if r["present"] and "iou" in r]
                d["refcoco_val_acc50"] = float(np.mean([r["iou"] >= 0.5 for r in single]))
                d["gref_N_acc"] = float((pr[~y] < 0.5).mean()); d["gref_T_acc"] = float((pr[y] >= 0.5).mean())
                d["gref_auroc"] = auroc(pr, y)
                if x == 1.0:
                    ref_g = d["gref_T_acc"]
                d["_pr"] = (pr, y)
            M[x] = d
        if ref_g is not None:
            for x, d in M.items():
                if "_pr" in d:
                    pr, y = d["_pr"]
                    t = np.quantile(pr[y], 1 - ref_g)
                    d["gref_N_acc_at_a1_recall"] = float((pr[~y] < t).mean())
        for d in M.values():
            d.pop("_pr", None)
        res[model] = M
        for x, d in M.items():
            print(model, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if not k.endswith("ci95")})
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
