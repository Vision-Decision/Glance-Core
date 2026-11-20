"""Prop. 1 two-knob rule (round-3 request), CPU only, from saved rows (outputs/ablation/<run>/rows*.json).

python ablation/twoknob_v2.py [--src outputs/ablation] [--out outputs/ablation_v2/twoknob]

Items: gRefCOCO-val-merged, 10,834 single-target present (with IoU) + 8,905 no-target (as selective_v2.py).
Cost model: answering an absent item costs c_H, answering a present item with IoU<0.5 costs c_W,
refusing a present item costs c_R, everything else 0. Reported: expected cost per item (lower is better).
Rules
  Glance (ours_neg25 / ours_neg05):
    plugin_2knob   Prop. 1 with the raw heads: answer iff (c_R-c_W+c_H) g + c_W g*k > c_H        (no tuning)
    plugin_gate    Prop. 1(a), ignores box quality: answer iff g > c_H/(c_R+c_H)                (no tuning)
    tuned_gate     best threshold on g                                         (tuned on the other half)
    tuned_gk       best threshold on s = g*k (Chow on s)                       (tuned on the other half)
    tuned_2knob    best linear rule  w g + (1-w) g*k > t, w in [0,1]           (tuned on the other half)
  AR heads (eta_AR = 1-P(none), kappa_AR = length-normalised box confidence of the constrained greedy box):
    greedy, seqmap (fixed operating points), tuned_pnone (threshold on 1-P(none)), tuned_margin (logit bias on
    "none" at step 1; = AR margin sweep), tuned_2knob_ar (same linear family on (eta_AR, eta_AR*kappa_AR)).
Tuning uses 2-fold cross-fitting over images (record index parity): the rule applied to one half is tuned on the
other, so every item is scored out-of-sample. 95% CIs: image-level bootstrap (1000) of the per-item cost.
Also: risk-coverage AURC of the linear scores w g + (1-w) g k for w in a grid (risk = errors incl. box errors).
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selective_v2 import load_rows, n_box_tokens, aurc

RUNS = ["ours_neg25", "ours_neg05", "ar_digits_neg25", "ar_hier_neg25", "ar_atomic100_neg25", "ar_atomic1000_neg25",
        "ar_digits_neg05"]
COSTS = [(1, r, w) for r in (0.5, 1.0, 2.0) for w in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)]   # (c_H, c_R, c_W)


def platt_fit(x, y):
    """1-D logistic regression p = sigmoid(a x + b) (L-BFGS on the log-loss, tiny L2 for stability)."""
    from scipy.optimize import minimize

    def f(w):
        z = w[0] * x + w[1]
        ll = np.logaddexp(0, z) - y * z
        p = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        g = np.array([((p - y) * x).mean(), (p - y).mean()]) + 1e-6 * w
        return ll.mean() + 5e-7 * (w @ w), g
    return minimize(f, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B").x


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def platt_crossfit(d, ar):
    """Cross-fitted (2-fold, image halves) Platt scaling of eta (target: present) and kappa (target: IoU>=.5 on
    present items). Glance features: logit(g), logit(kappa). AR features: eta <- margin = log max_c q(c) - log q(none);
    kappa <- log of the length-normalised box confidence."""
    fe = d["margin"] if ar else logit(d["eta"])
    fk = np.log(np.clip(d["kap"], 1e-12, 1)) if ar else logit(d["kap"])
    half = d["img"] % 2 == 0
    eta, kap = np.zeros(len(fe)), np.zeros(len(fe))
    for tr, te in ((half, ~half), (~half, half)):
        we = platt_fit(fe[tr], d["y"][tr].astype(float))
        trp = tr & d["y"]
        wk = platt_fit(fk[trp], d["good"][trp].astype(float))
        eta[te] = 1 / (1 + np.exp(-(we[0] * fe[te] + we[1])))
        kap[te] = 1 / (1 + np.exp(-(wk[0] * fk[te] + wk[1])))
    return eta, kap
WGRID = np.linspace(0, 1, 21)


def load(src, run, prep):
    rows = load_rows(src, run)["grefcoco_val_merged"]
    rec = []
    for i, l in enumerate(open(prep)):
        rec += [i] * len(json.loads(l)["questions"])
    assert len(rec) == len(rows)
    keep = [k for k, r in enumerate(rows) if (not r["present"]) or ("iou" in r)]
    R = [rows[k] for k in keep]
    img = np.array([rec[k] for k in keep])
    y = np.array([r["present"] for r in R])
    iou = np.array([r.get("iou", 0.0) for r in R])
    d = dict(y=y, good=y & (iou >= .5), img=img)
    if run.startswith("ours"):
        d["eta"] = np.array([r["p_present"] for r in R]); d["kap"] = np.array([r["p_iou50"] for r in R])
    else:
        pn = np.array([r["p_none"] for r in R]); lpb = np.array([r["lp_box"] for r in R])
        L = np.array([n_box_tokens(run, r["box"]) if r.get("box") else 4 for r in R])
        pmc = np.array([r.get("p_maxcoord_c", r["p_maxcoord"]) for r in R])
        lpf = np.log(np.clip(pmc, 1e-30, 1))
        d["eta"] = 1 - pn
        d["kap"] = np.exp((lpb - lpf) / np.maximum(L - 1, 1))
        d["margin"] = np.log(np.clip(pmc, 1e-30, 1)) - np.log(np.clip(pn, 1e-30, 1))   # answer iff margin + b > 0
        d["greedy"] = ~np.array([r.get("greedy_none_c", r["greedy_none"]) for r in R])
        d["seqmap"] = ~np.array([r["seqmap_none"] for r in R])
    return d


def cost_vec(acc, d, c):
    cH, cR, cW = c
    return np.where(acc, np.where(~d["y"], cH, np.where(d["good"], 0.0, cW)), np.where(d["y"], cR, 0.0))


def best_thr(score, d, c, idx):
    """threshold t minimising mean cost of (score > t) on items idx."""
    s = score[idx]
    o = np.argsort(-s, kind="stable")
    cH, cR, cW = c
    acc_cost = np.where(~d["y"][idx], cH, np.where(d["good"][idx], 0.0, cW))[o]
    ref_cost = np.where(d["y"][idx], cR, 0.0)[o]
    # accept top-k: cost = sum acc_cost[:k] + sum ref_cost[k:]
    tot = np.concatenate([[0], np.cumsum(acc_cost)]) + (ref_cost.sum() - np.concatenate([[0], np.cumsum(ref_cost)]))
    k = int(np.argmin(tot))
    ss = s[o]
    if k == 0:
        return ss[0] + 1e-9
    if k == len(ss):
        return ss[-1] - 1e-9
    return (ss[k - 1] + ss[k]) / 2


def crossfit(score_fn, d, c, family=None):
    """Returns accept vector; rules tuned on the other image half. family: list of score vectors (choose best)."""
    half = d["img"] % 2 == 0
    acc = np.zeros(len(d["y"]), bool)
    for tr, te in ((half, ~half), (~half, half)):
        it, ie = np.where(tr)[0], np.where(te)[0]
        cands = family if family is not None else [score_fn]
        best = None
        for sc in cands:
            t = best_thr(sc, d, c, it)
            cc = cost_vec(sc[it] > t, {k: v[it] for k, v in d.items() if isinstance(v, np.ndarray)}, c).mean()
            if best is None or cc < best[0]:
                best = (cc, sc, t)
        acc[ie] = best[1][ie] > best[2]
    return acc


def boot_ci(cv, img, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(img, return_inverse=True)
    s = np.bincount(inv, weights=cv); n = np.bincount(inv)
    vals = []
    for _ in range(B):
        b = rng.integers(0, len(u), len(u))
        vals.append(s[b].sum() / n[b].sum())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/ablation")
    ap.add_argument("--prep", default="data/prep/eval_grefcoco_val_merged.jsonl")
    ap.add_argument("--out", default="outputs/ablation_v2/twoknob")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = {r: load(a.src, r, a.prep) for r in RUNS if load_rows(a.src, r) is not None}
    for r, d in D.items():
        d["cal"] = platt_crossfit(d, not r.startswith("ours"))
    res = dict(costs={}, aurc_linear={})
    for run, d in D.items():
        g, k = d["eta"], d["kap"]
        res["aurc_linear"][run] = {f"{w:.2f}": aurc(w * g + (1 - w) * g * k, d["good"]) for w in WGRID}
    for c in COSTS:
        cH, cR, cW = c
        key = f"cH={cH},cR={cR},cW={cW}"
        out = {}
        for run, d in D.items():
            g, k = d["eta"], d["kap"]
            fam = [w * g + (1 - w) * g * k for w in WGRID]
            rules = {}
            ce, ck = d["cal"]
            rules["platt_plugin_2knob"] = (cR - cW + cH) * ce + cW * ce * ck > cH     # Prop. 1, calibrated, no tuning
            rules["platt_plugin_eta_only"] = ce > cH / (cR + cH)
            if run.startswith("ours"):
                rules["plugin_2knob"] = (cR - cW + cH) * g + cW * g * k > cH
                rules["plugin_gate"] = g > cH / (cR + cH)
                rules["tuned_gate"] = crossfit(g, d, c)
                rules["tuned_gk"] = crossfit(g * k, d, c)
                rules["tuned_2knob"] = crossfit(None, d, c, family=fam)
            else:
                rules["greedy"] = d["greedy"]; rules["seqmap"] = d["seqmap"]
                rules["tuned_pnone"] = crossfit(g, d, c)
                rules["tuned_margin"] = crossfit(d["margin"], d, c)
                rules["tuned_2knob_ar"] = crossfit(None, d, c, family=fam)
            o = {}
            for name, acc in rules.items():
                cv = cost_vec(acc, d, c)
                o[name] = dict(cost=float(cv.mean()), ci95=boot_ci(cv, d["img"]), coverage=float(acc.mean()),
                               false_accept=float((acc & ~d["y"]).sum() / (~d["y"]).sum()),
                               box_error_accepted=float((acc & d["y"] & ~d["good"]).sum() / max(1, (acc & d["y"]).sum())),
                               refusal_present=float((~acc & d["y"]).sum() / d["y"].sum()))
            o["_acc"] = rules
            if run.startswith("ours"):
                diff = cost_vec(rules["tuned_2knob"], d, c) - cost_vec(rules["tuned_gate"], d, c)
                o["delta_tuned_2knob_minus_gate"] = dict(mean=float(diff.mean()), ci95=boot_ci(diff, d["img"], seed=1))
            out[run] = o
        # Glance vs AR, paired on the same items (Platt-calibrated Prop. 1 plug-in for both; and best tuned rules)
        for g_run, ar_runs in (("ours_neg25", [r for r in D if r.startswith("ar_") and r.endswith("neg25")]),
                               ("ours_neg05", ["ar_digits_neg05"])):
            if g_run not in D:
                continue
            for ar in ar_runs:
                if ar not in D or not np.array_equal(D[g_run]["img"], D[ar]["img"]):
                    continue
                dg, da = D[g_run], D[ar]
                for rg, ra in (("platt_plugin_2knob", "platt_plugin_2knob"), ("tuned_2knob", "tuned_margin")):
                    ag = out[g_run]["_acc"][rg]; aa = out[ar]["_acc"][ra]
                    diff = cost_vec(ag, dg, c) - cost_vec(aa, da, c)
                    out[g_run][f"delta_vs_{ar}:{rg}-{ra}"] = dict(mean=float(diff.mean()), ci95=boot_ci(diff, dg["img"], seed=2),
                                                                rel=float(diff.mean() / cost_vec(aa, da, c).mean()))
        for o in out.values():
            o.pop("_acc", None)
        res["costs"][key] = out
        print(key, {r: {n: round(v["cost"], 4) for n, v in o.items() if isinstance(v, dict) and "cost" in v} for r, o in out.items()}, flush=True)
    json.dump(res, open(f"{a.out}/twoknob.json", "w"), indent=1)
    # plot: cost vs c_W at c_H = c_R = 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ws = [w for (_, r, w) in COSTS if r == 1.0]
    sty = {("ours_neg25", "tuned_2knob"): ("#2a78d6", "-", "Glance two-knob (tuned)"),
           ("ours_neg25", "plugin_2knob"): ("#2a78d6", ":", "Glance two-knob (plug-in, no tuning)"),
           ("ours_neg25", "tuned_gate"): ("#7aa9e6", "--", "Glance gate only (tuned)"),
           ("ours_neg25", "tuned_gk"): ("#9b59b6", "-.", "Glance g*kappa (tuned)"),
           ("ar_digits_neg25", "tuned_margin"): ("#eb6834", "--", "AR-digits margin sweep (tuned)"),
           ("ar_digits_neg25", "tuned_2knob_ar"): ("#eb6834", "-", "AR-digits two-knob (tuned)"),
           ("ar_digits_neg25", "seqmap"): ("#000000", ":", "AR-digits greedy-cand. MAP")}
    for (run, rule), (col, ls, lab) in sty.items():
        if run not in D:
            continue
        ys = [res["costs"][f"cH=1,cR=1.0,cW={w}"][run][rule]["cost"] for w in ws]
        axes[0].plot(ws, ys, ls, color=col, label=lab, lw=1.6, marker="o", ms=3)
    axes[0].set_xlabel("c_W (box-error cost), c_H = c_R = 1"); axes[0].set_ylabel("expected cost per item")
    axes[0].legend(fontsize=6.5); axes[0].grid(alpha=.3); axes[0].set_title("Prop. 1 rule vs one-score rules", loc="left", fontsize=9)
    for run, col in (("ours_neg25", "#2a78d6"), ("ours_neg05", "#7aa9e6"), ("ar_digits_neg25", "#eb6834")):
        if run in res["aurc_linear"]:
            v = res["aurc_linear"][run]
            axes[1].plot(WGRID, [v[f"{w:.2f}"] * 100 for w in WGRID], color=col, label=run, lw=1.6)
    axes[1].set_xlabel("w in score w*eta + (1-w)*eta*kappa  (1 = gate only, 0 = eta*kappa)")
    axes[1].set_ylabel("AURC (%)"); axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
    axes[1].set_title("risk-coverage AURC along the two-knob family", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{a.out}/twoknob.png", dpi=150)


if __name__ == "__main__":
    main()
