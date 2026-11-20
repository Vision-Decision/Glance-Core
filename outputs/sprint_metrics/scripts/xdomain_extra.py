"""Cross-domain rows (v0 and the generative base on ScreenSpot-Pro / VenusBench-GD) with Wilson CIs.

    python outputs/sprint_metrics/scripts/xdomain_extra.py <rows_dir> <out_json>

rows_dir: outputs/eval_v0_xdomain/rows (the main checkout holds the complete files; the worktree copy of
base_sspro_4000.jsonl is the stale 718-row version). Gated positive accuracy counts a refused positive as wrong,
as the official VenusBench-GD scorer does.
"""
import json, sys
import numpy as np
from scipy.stats import rankdata

D, OUT = sys.argv[1], sys.argv[2]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def refused(x):
    return bool(x["refused"]) if "refused" in x else x["p_present"] < .5


res = {}
for f in ["base_ss2_4000", "ours_ss2_640", "base_sspro_4000", "ours_sspro_640", "ours_sspro_4000", "base_venus_4000",
          "ours_venus_640", "ours_venus_4000"]:
    rr = [json.loads(l) for l in open(f"{D}/{f}.jsonl")]
    P = [x for x in rr if x["present"]]; A = [x for x in rr if not x["present"]]
    d = {"n": len(rr), "n_pos": len(P), "n_ref": len(A)}
    hp = [bool(x["hit"]) for x in P]
    d["pos_click"] = round(100 * np.mean(hp), 1); d["pos_ci"] = wilson(sum(hp), len(hp))
    if A:
        ra = [refused(x) for x in A]
        gp = [bool(x["hit"]) and not refused(x) for x in P]
        d.update(refusal=round(100 * np.mean(ra), 1), ref_ci=wilson(sum(ra), len(ra)),
                 pos_click_gated=round(100 * np.mean(gp), 1),
                 overall_micro_ungated=round(100 * (sum(hp) + sum(ra)) / len(rr), 1),
                 overall_micro_gated=round(100 * (sum(gp) + sum(ra)) / len(rr), 1),
                 ids_first_last=[rr[0]["id"], rr[-1]["id"]])
        if rr[0].get("p_present") is not None:
            p = np.array([x["p_present"] for x in rr]); y = np.array([x["present"] for x in rr]).astype(int)
            rk = rankdata(p); a, b = y.sum(), (1 - y).sum()
            d["auroc"] = round(100 * float((rk[y == 1].sum() - a * (a + 1) / 2) / (a * b)), 1)
    res[f] = d
    print(f, json.dumps(d))
json.dump(res, open(OUT, "w"), indent=1)
