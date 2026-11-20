"""Experiment 4: selective-prediction reanalysis of the controlled heads (CPU only, from saved rows).

python ablation/selective_v2.py [--src outputs/ablation] [--out outputs/ablation_v2/selective]

Items: gRefCOCO-val-merged single-target present (= RefCOCO val, 10,834; carries IoU) + no-target (8,905).
Decision = accept (output the box) or refuse. Error = accept on an absent item (false accept) or accept on a
present item with IoU < 0.5 (box error). Refusing is never an error but lowers coverage.
coverage = accepted / all items; selective risk = errors / accepted.  AURC = area under risk-coverage.
Scores (higher = accept first):
  ours:  gate          p_present                         (presence only)
         gate*iou      p_present * p_iou50               (= pi*c, P(present and IoU>=.5))
  AR:    1-P(none)     1 - q(t1 = none)                  (presence only)
         P(s_hat)      q(s_hat), joint prob. of the box-forced greedy sequence (incl. first token)
         (1-P(none))*conf  (1 - q(none)) * exp(mean log q of the box tokens after the first) (length-normalised
                       localisation confidence of the decoded box)
Operating points: ours gate>=.5, gate*iou>=.5; AR greedy (constrained), P(none)<.5, seq-MAP proxy.
Also: Acc@0.5 of present targets refused vs accepted by seq-MAP; mean IoU / Acc@.5/.75/.9 for every head.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = ["ours_neg25", "ar_digits_neg25", "ar_hier_neg25", "ar_atomic100_neg25", "ar_atomic1000_neg25",
        "ours_neg05", "ar_digits_neg05"]
LABEL = {"ours_neg25": "Gate", "ar_digits_neg25": "AR-digits", "ar_hier_neg25": "AR-hier",
         "ar_atomic100_neg25": "AR-atomic-100", "ar_atomic1000_neg25": "AR-atomic-1000",
         "ours_neg05": "Gate (5% neg)", "ar_digits_neg05": "AR-digits (5% neg)"}
COLOR = {"ours": "#2a78d6", "ar_digits": "#eb6834", "ar_hier": "#1baf7a", "ar_atomic1000": "#eda100",
         "ar_atomic100": "#e87ba4"}


def color(run):
    for k in sorted(COLOR, key=len, reverse=True):
        if run.startswith(k):
            return COLOR[k]


def load_rows(src, run):
    d = f"{src}/{run}"
    for f in ("rows_c.json", "rows.json"):
        if os.path.exists(f"{d}/{f}"):
            return json.load(open(f"{d}/{f}"))
    return None


def n_box_tokens(run, box):
    """Answer length (excluding <box_end>) of the decoded box for each AR format."""
    if "digits" in run:
        return sum(len(str(int(round(v * 1000)))) for v in box) + 3
    if "hier" in run:
        return 8
    return 4


def items(rows, run):
    R = [r for r in rows["grefcoco_val_merged"] if (not r["present"]) or ("iou" in r)]
    ar = not run.startswith("ours")
    y_abs = np.array([not r["present"] for r in R])
    iou = np.array([r.get("iou", 0.0) for r in R])
    ok_if_accept = (~y_abs) & (iou >= 0.5)
    S, ops = {}, {}
    if not ar:
        pp = np.array([r["p_present"] for r in R]); pi = np.array([r["p_iou50"] for r in R])
        S["gate"] = pp; S["gate*iou"] = pp * pi
        ops["gate>=.5"] = pp >= .5; ops["gate*iou>=.5"] = pp * pi >= .5
    else:
        pn = np.array([r["p_none"] for r in R])
        lpb = np.array([r["lp_box"] for r in R])
        S["1-P(none)"] = 1 - pn
        S["P(s_hat)"] = np.exp(lpb)
        L = np.array([n_box_tokens(run, r["box"]) if r.get("box") else 4 for r in R])
        lp_first = np.log(np.clip(np.array([r.get("p_maxcoord_c", r["p_maxcoord"]) for r in R]), 1e-30, 1))
        conf = np.exp((lpb - lp_first) / np.maximum(L - 1, 1))
        S["(1-P(none))*conf"] = (1 - pn) * conf
        ops["greedy"] = ~np.array([r.get("greedy_none_c", r["greedy_none"]) for r in R])
        ops["P(none)<.5"] = pn < .5
        ops["seq-MAP proxy"] = ~np.array([r["seqmap_none"] for r in R])
    return R, ok_if_accept, y_abs, iou, S, ops


def rc_curve(score, ok):
    o = np.argsort(-score, kind="stable")
    okc = np.cumsum(ok[o])
    n = np.arange(1, len(o) + 1)
    risk = 1 - okc / n
    cov = n / len(o)
    return cov, risk


def aurc(score, ok):
    cov, risk = rc_curve(score, ok)
    return float(np.trapezoid(risk, cov) if hasattr(np, "trapezoid") else np.trapz(risk, cov))


def risk_at_cov(score, ok, c):
    cov, risk = rc_curve(score, ok)
    i = min(len(cov) - 1, max(0, int(round(c * len(cov))) - 1))
    return float(risk[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/ablation")
    ap.add_argument("--out", default="outputs/ablation_v2/selective")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    res, curves = {}, {}
    for run in RUNS:
        rows = load_rows(a.src, run)
        if rows is None:
            continue
        R, ok, y_abs, iou, S, ops = items(rows, run)
        N = len(R)
        boots = [rng.integers(0, N, N) for _ in range(a.boot)]
        r = dict(n=N, n_present=int((~y_abs).sum()), n_absent=int(y_abs.sum()), scores={}, operating_points={})
        for name, sc in S.items():
            au = aurc(sc, ok)
            bs = [aurc(sc[b], ok[b]) for b in boots]
            r["scores"][name] = dict(aurc=au, aurc_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
            cov, risk = rc_curve(sc, ok)
            k = np.linspace(0, len(cov) - 1, 400).astype(int)
            curves[f"{run}/{name}"] = dict(coverage=cov[k].tolist(), risk=risk[k].tolist())
        for name, acc in ops.items():
            cov = float(acc.mean()); risk = float(1 - ok[acc].mean()) if acc.any() else 0.0
            # risk of every score of this head at the same coverage
            matched = {sn: risk_at_cov(sc, ok, cov) for sn, sc in S.items()}
            r["operating_points"][name] = dict(
                coverage=cov, selective_risk=risk,
                false_accept_rate_absent=float(acc[y_abs].mean()), refusal_present=float((~acc[~y_abs]).mean()),
                box_error_rate_accepted_present=float((iou[acc & ~y_abs] < .5).mean()),
                risk_of_scores_at_same_coverage=matched)
        pres = ~y_abs
        r["iou"] = dict(mean_iou=float(iou[pres].mean()), acc50=float((iou[pres] >= .5).mean()),
                        acc75=float((iou[pres] >= .75).mean()), acc90=float((iou[pres] >= .9).mean()))
        # all three RefCOCO splits
        allp = [x for s in ("grefcoco_val_merged", "refcoco_testA", "refcoco_testB") for x in rows[s] if x["present"] and "iou" in x]
        ii = np.array([x["iou"] for x in allp])
        r["iou_val_testA_testB"] = dict(n=len(ii), mean_iou=float(ii.mean()), acc50=float((ii >= .5).mean()),
                                        acc75=float((ii >= .75).mean()), acc90=float((ii >= .9).mean()))
        for s in ("refcoco_testA", "refcoco_testB"):
            jj = np.array([x["iou"] for x in rows[s] if x["present"] and "iou" in x])
            r[f"iou_{s}"] = dict(mean_iou=float(jj.mean()), acc50=float((jj >= .5).mean()),
                                 acc75=float((jj >= .75).mean()), acc90=float((jj >= .9).mean()))
        if "seq-MAP proxy" in ops:
            refused = pres & ~ops["seq-MAP proxy"]; accepted = pres & ops["seq-MAP proxy"]
            r["seqmap_refused_present"] = dict(
                n_refused=int(refused.sum()), acc50_refused=float((iou[refused] >= .5).mean()),
                acc50_accepted=float((iou[accepted] >= .5).mean()), mean_iou_refused=float(iou[refused].mean()),
                mean_iou_accepted=float(iou[accepted].mean()),
                parse_ok_share_of_refused=float(np.mean([R[i].get("parse") == "box" for i in np.where(refused)[0]])))
        res[run] = r
        print(run, json.dumps({k: v for k, v in r.items() if k not in ("scores",)})[:600], flush=True)
    json.dump(res, open(f"{a.out}/selective.json", "w"), indent=1)
    json.dump(curves, open(f"{a.out}/risk_coverage_curves.json", "w"))
    # ---------------- plot: one panel per neg25 head + an overlay of the best score per head
    main = [r for r in RUNS if r in res and r.endswith("neg25")]
    fig, axes = plt.subplots(1, len(main) + 1, figsize=(3.6 * (len(main) + 1), 3.6), sharey=True)
    styles = ["-", "--", ":"]
    for ax, run in zip(axes, main):
        for j, name in enumerate(res[run]["scores"]):
            c = curves[f"{run}/{name}"]
            ax.plot(c["coverage"], np.array(c["risk"]) * 100, styles[j % 3], color=color(run), lw=1.6,
                    label=f"{name} (AURC {res[run]['scores'][name]['aurc']*100:.1f})")
        for k, (name, op) in enumerate(res[run]["operating_points"].items()):
            ax.plot(op["coverage"], op["selective_risk"] * 100, "osD^"[k], color="#333333", ms=5, mfc="none",
                    label=f"{name}: cov {op['coverage']*100:.1f}, risk {op['selective_risk']*100:.1f}")
        ax.set_title(LABEL[run], loc="left", fontsize=10)
        ax.set_xlabel("coverage")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_ylim(0, 45); ax.set_xlim(0, 1)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("selective risk (%)\n(false accept on absent or IoU<0.5)")
    ax = axes[-1]
    for run in main:
        best = min(res[run]["scores"], key=lambda n: res[run]["scores"][n]["aurc"])
        c = curves[f"{run}/{best}"]
        ax.plot(c["coverage"], np.array(c["risk"]) * 100, color=color(run), lw=1.6, label=f"{LABEL[run]}: {best}")
    ax.set_title("best score per head", loc="left", fontsize=10); ax.set_xlabel("coverage"); ax.legend(fontsize=6.5)
    ax.grid(alpha=.3)
    fig.suptitle("Risk-coverage on gRefCOCO-val-merged (RefCOCO val present + gRefCOCO no-target, n=19,739)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{a.out}/risk_coverage.png", dpi=150)


if __name__ == "__main__":
    main()
