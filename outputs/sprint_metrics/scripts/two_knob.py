"""Prop. 1 two-knob rule vs presence-only rule, realized cost on gRefCOCO val merged (single-target + no-target).

    python outputs/sprint_metrics/scripts/two_knob.py [repo_root]

Plug-in rules, nothing tuned on the evaluation data:
  gate-only  (Prop. 1a):  answer iff c_H (1 - g) < c_R g
  two-knob   (Prop. 1)  :  answer iff c_H (1 - g) + c_W g (1 - k) < c_R g
  AR plug-in (Prop. 1a on 1 - q(none)); AR two-knob with the margin is not a probability, so it is not run.
Costs: absent answered c_H; present answered with IoU<.5 c_W; present refused c_R. c_H = c_R = 1, c_W varied.
Absent rate varied by reweighting absent items (importance weights; same items), with the prior correction of
Remark 1 applied to g in every rule (training absent rate 25%).
Paired bootstrap (1,000, seed 0) of mean-cost differences.
"""
import json, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
O = f"{ROOT}/outputs/ablation"
rng = np.random.default_rng(0)


def rows(r, f):
    return [x for x in json.load(open(f"{O}/{r}/{f}"))["grefcoco_val_merged"] if (not x["present"]) or "iou" in x]


G = rows("ours_neg25", "rows.json")
pres = np.array([bool(x["present"]) for x in G]); ok = np.array([x.get("iou", 0) >= .5 for x in G]) & pres
g0 = np.clip(np.array([x["p_present"] for x in G]), 1e-9, 1 - 1e-9); k = np.array([x["p_iou50"] for x in G])
AR = {r: rows(r, "rows_c.json") for r in ["ar_digits_neg25", "ar_hier_neg25", "ar_atomic1000_neg25", "ar_atomic100_neg25"]}
ARs = {r: (np.clip(np.array([1 - x["p_none"] for x in v]), 1e-9, 1 - 1e-9),
           np.array([x.get("iou", 0) >= .5 for x in v]) & pres) for r, v in AR.items()}


def shift(p, pi_te, pi_tr=.25):  # prior correction of the presence probability to absent rate pi_te
    lo = np.log(p / (1 - p)) + np.log((pi_tr / (1 - pi_tr)) * ((1 - pi_te) / pi_te)); return 1 / (1 + np.exp(-lo))


def cost(ans, pres_, ok_, cW, w):
    c = np.where(ans, np.where(~pres_, 1.0, np.where(ok_, 0.0, cW)), np.where(pres_, 1.0, 0.0))
    return c, w


out = {}
n_abs, n_pres = (~pres).sum(), pres.sum()
for pi in (0.05, 0.25, 0.45):
    w = np.where(pres, (1 - pi) / n_pres, pi / n_abs) * len(G)   # reweight to absent rate pi
    for cW in (0.0, 0.5, 1.0, 2.0, 5.0):
        g = shift(g0, pi)
        rules = {"gate_only": (1 - g) < g,                                   # c_H = c_R = 1
                 "two_knob": (1 - g) + cW * g * (1 - k) < g,
                 "always_answer": np.ones(len(G), bool)}
        C = {nm: cost(a, pres, ok, cW, w)[0] for nm, a in rules.items()}
        for r, (p, okr) in ARs.items():
            pr = shift(p, pi)
            C[f"{r}_thresh"] = cost((1 - pr) < pr, pres, okr, cW, w)[0]
        res = {nm: float((c * w).mean()) for nm, c in C.items()}
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(G), len(G)); ww = w[i]
            bs.append([(C["two_knob"][i] * ww).mean() - (C["gate_only"][i] * ww).mean()] +
                      [(C["gate_only"][i] * ww).mean() - (C[f"{r}_thresh"][i] * ww).mean() for r in ARs])
        bs = np.array(bs)
        res["two_knob_minus_gate_only"] = [float(bs[:, 0].mean()), float(np.quantile(bs[:, 0], .025)), float(np.quantile(bs[:, 0], .975))]
        for j, r in enumerate(ARs):
            res[f"gate_only_minus_{r}"] = [float(bs[:, j + 1].mean()), float(np.quantile(bs[:, j + 1], .025)), float(np.quantile(bs[:, j + 1], .975))]
        res["two_knob_coverage"] = float((rules["two_knob"] * w).mean()); res["gate_only_coverage"] = float((rules["gate_only"] * w).mean())
        out[f"pi{pi}_cW{cW}"] = res
json.dump(out, open(f"{ROOT}/outputs/sprint_metrics/two_knob.json", "w"), indent=1)
for key, v in out.items():
    print(key, "gate", round(v["gate_only"], 4), "2knob", round(v["two_knob"], 4), "diff", [round(x, 4) for x in v["two_knob_minus_gate_only"]],
          "| gate-AR(digits)", [round(x, 4) for x in v["gate_only_minus_ar_digits_neg25"]], "cov", round(v["two_knob_coverage"], 3), round(v["gate_only_coverage"], 3))
