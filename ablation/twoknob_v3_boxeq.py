"""Two-knob cost with box accuracy held equal.

Uses the calibrated (eta, kappa) of twoknob_v3 (all protocols) and re-scores the cost only on items where Glance's and
AR-digits' boxes agree in correctness (all absent items + present items with good_g == good_a), so both systems face the
same box errors and any remaining gap comes from the decision scores. Run from the repo root:
    uv run --with numpy python -m ablation.twoknob_v3_boxeq
"""
import json
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import twoknob_v3 as T
from twoknob_v2 import cost_vec, boot_ci

OUT = "outputs/ablation_v3/ar_iou"
I = dict(np.load(f"{OUT}/items.npz"))
y, img = I["t_y"], I["t_img"]
S = (~y) | (I["t_good_g"] == I["t_good_a"])
cal, _ = T.calibrate(I)
res = dict(n_subset=int(S.sum()), n_present_subset=int((S & y).sum()),
           box_error_subset=float(1 - I["t_good_g"][S & y].mean()), n_dropped_present=int((y & ~S).sum()))
for pr, C in cal.items():
    tp = {}
    for cW in T.CW:
        acc = {}
        for m in ("glance", "ar", "ar+heldout_full"):
            eta, kap = C[m]
            g = I["t_good_g"] if m == "glance" else I["t_good_a"]
            acc[m] = cost_vec(T.two_knob(eta, kap, cW), dict(y=y, good=g), (1.0, 1.0, cW))[S]
        diff = acc["glance"] - acc["ar+heldout_full"]
        tp[f"cW={cW}"] = {**{f"{m}:two_knob": float(v.mean()) for m, v in acc.items()},
                          "delta_glance_minus_ar+heldout_full": dict(mean=float(diff.mean()), ci95=boot_ci(diff, img[S], seed=2),
                                                                     rel=float(diff.mean() / acc["ar+heldout_full"].mean()))}
    res[pr] = tp
json.dump(res, open(f"{OUT}/boxeq.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("n_subset", "n_present_subset", "box_error_subset", "n_dropped_present")}))
for pr in cal:
    for cW, r in res[pr].items():
        d = r["delta_glance_minus_ar+heldout_full"]
        print(pr, cW, {k: round(v * 100, 2) for k, v in r.items() if k.endswith("two_knob")}, round(d["mean"] * 100, 2), [round(x * 100, 2) for x in d["ci95"]], round(d["rel"] * 100, 1))
