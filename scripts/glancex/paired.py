"""Paired, image-clustered bootstrap of the RefCOCO 8-split mean Acc@0.5 between two row dumps (CPU).

  python scripts/glancex/paired.py ROWS_A ROWS_B [--B 2000] [--out x.json]

Rows are matched on (set, image_id, qi, text). Each split's images are resampled with replacement (B times); the
statistic is mean_over_splits(acc_A) - mean_over_splits(acc_B), as in the usual 8-split average.
"""
import argparse, gzip, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import REFCOCO, iou  # noqa: E402


def hits(d, s):
    out = {}
    for l in gzip.open(os.path.join(d, s + ".jsonl.gz"), "rt"):
        r = json.loads(l)
        out[(r["image_id"], r["qi"], r["text"])] = (r["image_id"], float(iou(r["box"], r["gt"][0]) >= 0.5))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    per, boot = {}, np.zeros(a.B)
    for s in REFCOCO:
        ha, hb = hits(a.a, s), hits(a.b, s)
        keys = sorted(set(ha) & set(hb), key=str)
        img = np.array([ha[k][0] for k in keys]); da = np.array([ha[k][1] for k in keys]); db = np.array([hb[k][1] for k in keys])
        u, inv = np.unique(img, return_inverse=True)
        sa, sb, n = np.bincount(inv, da), np.bincount(inv, db), np.bincount(inv)
        per[s] = dict(n=len(keys), unmatched=len(ha) + len(hb) - 2 * len(keys), a=float(da.mean()), b=float(db.mean()))
        for i in range(a.B):
            idx = rng.integers(0, len(u), len(u))
            boot[i] += (sa[idx].sum() / n[idx].sum() - sb[idx].sum() / n[idx].sum()) / len(REFCOCO)
    diff = float(np.mean([per[s]["a"] - per[s]["b"] for s in REFCOCO]))
    res = dict(a=a.a, b=a.b, mean_a=float(np.mean([per[s]["a"] for s in REFCOCO])), mean_b=float(np.mean([per[s]["b"] for s in REFCOCO])),
               diff=diff, ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], per_split=per)
    print(json.dumps({k: v for k, v in res.items() if k != "per_split"}))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
