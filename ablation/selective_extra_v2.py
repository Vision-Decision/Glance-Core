"""Experiment 5 (extra items), CPU only, from saved rows of outputs/ablation/<run>/rows*.json.

python ablation/selective_extra_v2.py [--src outputs/ablation] [--out outputs/ablation_v2/selective]

Same item set as ablation/selective_v2.py (gRefCOCO-val-merged: 10,834 single-target present + 8,905 no-target).
  (a) positive-only risk-coverage: present items only, error = IoU < 0.5 (box error). Scores: ours gate, gate*iou,
      iou head alone; AR 1-P(none), P(s_hat), length-normalised box confidence.
  (b) AR first-token margin sweep: a logit bias b added to "none" at step 1 of constrained greedy. Refuse iff
      log q(none) + b > log max_c q(c) (max over grammar-valid first coordinate tokens; later tokens are unaffected).
      For each b: N-acc (absent refused), present refusal, coverage, selective risk (incl. box errors). Compared with the
      gate threshold at the same N-acc.
  (c) MAP-refused present targets (seq-MAP proxy): n, mean IoU, Acc@0.5 of refused vs accepted, for every AR head.
  (d) AURC(g) vs AURC(g*kappa) check for the gate heads: values at full precision, rank agreement of the two orderings,
      Kendall-tau-like pair disagreement on a sample, AURC under kappa-only / random kappa / oracle, and the
      decomposition of the error set (absent false accepts vs box errors).
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selective_v2 import RUNS, LABEL, color, load_rows, items, rc_curve, aurc, n_box_tokens  # noqa


def pos_only(R, run, iou):
    pres = np.array([r["present"] for r in R])
    ok = iou[pres] >= .5
    S = {}
    if run.startswith("ours"):
        pp = np.array([r["p_present"] for r in R])[pres]; pi = np.array([r["p_iou50"] for r in R])[pres]
        S = {"gate": pp, "gate*iou": pp * pi, "iou": pi}
    else:
        pn = np.array([r["p_none"] for r in R])[pres]
        lpb = np.array([r["lp_box"] for r in R])[pres]
        L = np.array([n_box_tokens(run, r["box"]) if r.get("box") else 4 for r in R])[pres]
        lpf = np.log(np.clip(np.array([r.get("p_maxcoord_c", r["p_maxcoord"]) for r in R]), 1e-30, 1))[pres]
        S = {"1-P(none)": 1 - pn, "P(s_hat)": np.exp(lpb), "conf": np.exp((lpb - lpf) / np.maximum(L - 1, 1))}
    return ok, S


def auroc(score, pos):
    o = np.argsort(score, kind="stable")
    r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    n1 = pos.sum(); n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/ablation")
    ap.add_argument("--out", default="outputs/ablation_v2/selective")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    res = dict(positive_only={}, margin_sweep={}, map_refused={}, aurc_check={})
    data = {}
    for run in RUNS:
        rows = load_rows(a.src, run)
        if rows is not None:
            data[run] = items(rows, run)
    # (a) positive-only
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for run, (R, ok, y_abs, iou, S, ops) in data.items():
        okp, SP = pos_only(R, run, iou)
        res["positive_only"][run] = dict(n=int(len(okp)), base_error=float(1 - okp.mean()), scores={
            k: dict(aurc=aurc(v, okp), auroc_iou50=auroc(v, okp)) for k, v in SP.items()})
        if run.endswith("neg25"):
            best = min(SP, key=lambda k: aurc(SP[k], okp))
            for j, (k, v) in enumerate(SP.items()):
                cov, risk = rc_curve(v, okp)
                ax = axes[0] if run.startswith("ours") or k == best else None
                if ax is not None:
                    ax.plot(cov, risk * 100, ["-", "--", ":"][j % 3] if run.startswith("ours") else "-",
                            color=color(run), lw=1.4, label=f"{LABEL[run]}: {k} ({aurc(v, okp)*100:.2f})")
    axes[0].set_title("present targets only: risk = IoU<0.5", loc="left", fontsize=9)
    axes[0].set_xlabel("coverage"); axes[0].set_ylabel("risk (%)"); axes[0].legend(fontsize=6); axes[0].grid(alpha=.3)
    axes[0].set_ylim(0, 8)
    # (b) margin sweep
    ours = data.get("ours_neg25")
    for run, (R, ok, y_abs, iou, S, ops) in data.items():
        if run.startswith("ours"):
            continue
        pn = np.array([r.get("p_none_c", r["p_none"]) for r in R])
        pc = np.array([r.get("p_maxcoord_c", r["p_maxcoord"]) for r in R])
        m = np.log(np.clip(pn, 1e-30, 1)) - np.log(np.clip(pc, 1e-30, 1))       # refuse iff m + b > 0
        bs = np.linspace(-12, 12, 97)
        pts = []
        for b in bs:
            acc = ~(m + b > 0)
            pts.append(dict(bias=float(b), N_acc=float((~acc[y_abs]).mean()), refusal_present=float((~acc[~y_abs]).mean()),
                            coverage=float(acc.mean()), risk=float(1 - ok[acc].mean()) if acc.any() else 0.0))
        out = dict(points=pts, aurc_margin=aurc(-m, ok))
        gref = "ours_neg25" if run.endswith("neg25") else "ours_neg05"
        if gref in data:
            Rg, okg, yg, _, Sg, _ = data[gref]
            gp = Sg["gate"]
            cmp = []
            for target in (0.80, 0.90, 0.95, 0.99):
                # AR: smallest bias reaching N-acc >= target; gate: threshold reaching the same N-acc
                p = next((p for p in pts if p["N_acc"] >= target), None)
                if p is None:
                    continue
                thr = np.quantile(gp[yg], p["N_acc"])
                accg = gp > thr
                cmp.append(dict(N_acc=p["N_acc"], ar_bias=p["bias"], ar_refusal_present=p["refusal_present"],
                                ar_risk=p["risk"], gate_refusal_present=float((~accg[~yg]).mean()),
                                gate_risk=float(1 - okg[accg].mean())))
            out["vs_gate_at_same_N_acc"] = cmp
        res["margin_sweep"][run] = out
        if run.endswith("neg25"):
            axes[1].plot([p["N_acc"] * 100 for p in pts], [p["refusal_present"] * 100 for p in pts], color=color(run),
                         lw=1.4, label=LABEL[run] + " (bias on none)")
    if ours is not None:
        Rg, okg, yg, _, Sg, _ = ours
        gp = Sg["gate"]
        th = np.quantile(gp, np.linspace(0, 1, 400))
        axes[1].plot([(gp[yg] < t).mean() * 100 for t in th], [(gp[~yg] < t).mean() * 100 for t in th],
                     color=color("ours"), lw=1.8, label="Gate threshold")
    axes[1].set_xlim(50, 100); axes[1].set_ylim(0, 30)
    axes[1].set_xlabel("N-acc on absent (%)"); axes[1].set_ylabel("refusal of present (%)")
    axes[1].set_title("AR logit bias on 'none' vs gate threshold", loc="left", fontsize=9)
    axes[1].legend(fontsize=6); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{a.out}/positive_only_and_margin.png", dpi=150)
    # (c) MAP-refused
    for run, (R, ok, y_abs, iou, S, ops) in data.items():
        if "seq-MAP proxy" not in ops:
            continue
        pres = ~y_abs
        ref = pres & ~ops["seq-MAP proxy"]; acc = pres & ops["seq-MAP proxy"]
        res["map_refused"][run] = dict(n_present=int(pres.sum()), n_refused=int(ref.sum()),
                                       refused_share=float(ref.sum() / pres.sum()),
                                       mean_iou_refused=float(iou[ref].mean()), acc50_refused=float((iou[ref] >= .5).mean()),
                                       mean_iou_accepted=float(iou[acc].mean()), acc50_accepted=float((iou[acc] >= .5).mean()))
    # (d) AURC(g) vs AURC(g*kappa)
    for run in ("ours_neg25", "ours_neg05"):
        if run not in data:
            continue
        R, ok, y_abs, iou, S, ops = data[run]
        g, gk = S["gate"], S["gate*iou"]
        k = gk / np.maximum(g, 1e-12)
        oracle = ok.astype(float) + 1e-6 * g
        pres = ~y_abs
        n_err_abs = int(y_abs.sum()); n_err_box = int((pres & ~ok).sum())
        o1, o2 = np.argsort(-g, kind="stable"), np.argsort(-gk, kind="stable")
        r1 = np.empty(len(g)); r1[o1] = np.arange(len(g)); r2 = np.empty(len(g)); r2[o2] = np.arange(len(g))
        sp = float(np.corrcoef(r1, r2)[0, 1])
        i, j = rng.integers(0, len(g), 200000), rng.integers(0, len(g), 200000)
        disc = float(np.mean(np.sign(g[i] - g[j]) != np.sign(gk[i] - gk[j])))
        kperm = rng.permutation(k)
        # risk differences along the curve
        c1, rk1 = rc_curve(g, ok); c2, rk2 = rc_curve(gk, ok)
        res["aurc_check"][run] = dict(
            aurc_g=aurc(g, ok), aurc_gk=aurc(gk, ok), diff=aurc(gk, ok) - aurc(g, ok),
            aurc_presence_only_error_g=aurc(g, ~y_abs), aurc_presence_only_error_gk=aurc(gk, ~y_abs),
            aurc_kappa_only=aurc(k, ok), aurc_g_times_random_kappa=aurc(g * kperm, ok), aurc_oracle=aurc(oracle, ok),
            aurc_random=float(1 - ok.mean()),
            n_items=len(g), n_absent=n_err_abs, n_present_box_error=n_err_box,
            spearman_rank_g_vs_gk=sp, pair_order_disagreement=disc,
            frac_items_with_gate_above_0_99=float((g > .99).mean()), frac_kappa_above_0_9=float((k > .9).mean()),
            max_abs_risk_diff_along_curve=float(np.max(np.abs(rk1 - rk2))),
            risk_at_cov={str(c): [float(rk1[int(c * len(g)) - 1]), float(rk2[int(c * len(g)) - 1])]
                         for c in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65)},
            positive_only_aurc=dict(g=aurc(g[pres], ok[pres]), gk=aurc(gk[pres], ok[pres]), kappa=aurc(k[pres], ok[pres])))
    json.dump(res, open(f"{a.out}/selective_extra.json", "w"), indent=1)
    print(json.dumps(res["aurc_check"], indent=1))
    print(json.dumps(res["map_refused"], indent=1))
    print(json.dumps({r: v["scores"] for r, v in res["positive_only"].items()}, indent=1))
    print(json.dumps({r: v.get("vs_gate_at_same_N_acc") for r, v in res["margin_sweep"].items()}, indent=1))


if __name__ == "__main__":
    main()
