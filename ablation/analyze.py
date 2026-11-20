"""Tables + plots from outputs/ablation/<run>/{eval.json,rows.json}.  python ablation/analyze.py [outputs/ablation]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "outputs/ablation"
RUNS = ["ours_neg25", "ar_digits_neg25", "ar_hier_neg25", "ar_atomic1000_neg25", "ar_atomic100_neg25",
        "ours_neg05", "ar_digits_neg05"]
LABEL = {"ours_neg25": "Ours (gate+ordinal)", "ar_digits_neg25": "AR-digits", "ar_hier_neg25": "AR-hier (32x32)",
         "ar_atomic1000_neg25": "AR-atomic 1000", "ar_atomic100_neg25": "AR-atomic 100",
         "ours_neg05": "Ours, 5% neg", "ar_digits_neg05": "AR-digits, 5% neg"}
COLOR = {"ours": "#2a78d6", "ar_digits": "#eb6834", "ar_hier": "#1baf7a", "ar_atomic1000": "#eda100",
         "ar_atomic100": "#e87ba4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "legend.frameon": False})


def color(run):
    for k in sorted(COLOR, key=len, reverse=True):
        if run.startswith(k):
            return COLOR[k]


def load(run):
    """eval/rows: unconstrained greedy decoding; eval_c/rows_c: grammar-constrained decoding (AR only)."""
    d = f"{ROOT}/{run}"
    if not os.path.exists(f"{d}/eval.json"):
        return None, None
    ev = json.load(open(f"{d}/eval.json"))
    rows = json.load(open(f"{d}/rows.json")) if os.path.exists(f"{d}/rows.json") else None
    if os.path.exists(f"{d}/eval_c.json"):
        ev["sets_c"] = json.load(open(f"{d}/eval_c.json"))["sets"]
    if os.path.exists(f"{d}/rows_c.json"):
        rows = json.load(open(f"{d}/rows_c.json"))   # superset of fields (incl. *_c presence stats)
    lat = f"{d}/latency.json"
    if os.path.exists(lat):
        ev["latency"] = json.load(open(lat)).get("latency")
    return ev, rows


def present_rows(rows):
    out = []
    for s in ["grefcoco_val_merged", "refcoco_testA", "refcoco_testB"]:
        out += [r for r in rows.get(s, []) if r["present"] and "iou" in r]
    return out


def main():
    E, R = {}, {}
    for r in RUNS:
        e, rows = load(r)
        if e:
            E[r], R[r] = e, rows
    # ------------------------------------------------------------------ main table
    tab = []
    for r, e in E.items():
        U = e["sets"]                     # unconstrained
        S = e.get("sets_c", U)            # constrained if available (AR), else same
        ar = e["variant"] != "ours"
        g, gu = S.get("grefcoco_val_merged", {}), U.get("grefcoco_val_merged", {})
        v, vu = S.get("refcoco_val", {}), U.get("refcoco_val", {})
        rule = "greedy" if ar else "gate"
        row = dict(run=r, variant=e["variant"], decoding=("constrained" if "sets_c" in e else "unconstrained") if ar else "-",
                   refcoco_val=v.get("acc50"), testA=S.get("refcoco_testA", {}).get("acc50"),
                   testB=S.get("refcoco_testB", {}).get("acc50"),
                   N_acc=gu.get(f"N_acc_{rule}"), refusal_present_val=vu.get(f"refusal_present_{rule}"),
                   auroc=g.get("auroc_presence"), ece=g.get("ece_presence"))
        if ar:
            row.update(N_acc_greedy_c=g.get("N_acc_greedy_c"), N_acc_thresh=gu.get("N_acc_thresh"),
                       N_acc_seqmap=g.get("N_acc_seqmap"),
                       refusal_greedy_c_val=v.get("refusal_present_greedy_c"),
                       refusal_thresh_val=vu.get("refusal_present_thresh"),
                       refusal_seqmap_val=v.get("refusal_present_seqmap"),
                       acc50_val_unconstrained=vu.get("acc50"), parse_fail_val_unconstrained=vu.get("parse_fail"))
        if R.get(r):
            rows = R[r]["grefcoco_val_merged"]
            sc = np.array([x["p_present"] if not ar else 1 - x["p_none"] for x in rows])
            pres = np.array([x["present"] and "iou" in x for x in rows]); absn = np.array([not x["present"] for x in rows])
            for fr in (0.005, 0.01, 0.02):   # N-acc when the threshold refuses exactly fr of RefCOCO-val present targets
                thr = np.quantile(sc[pres], fr)
                row[f"N_acc@{fr*100:g}%refusal"] = float((sc[absn] < thr).mean())
            if "neg05" in r:
                # post-hoc prior (logit) correction 5% -> 25% absent: logit' = logit - log(odds_present(5%)/odds_present(25%))
                shift = np.log((0.95 / 0.05) / (0.75 / 0.25))
                lo = np.log(np.clip(sc, 1e-12, 1 - 1e-12)) - np.log(np.clip(1 - sc, 1e-12, 1))
                say = lo - shift >= 0
                row["N_acc_priorcorr"] = float((~say[absn]).mean())
                row["refusal_val_priorcorr"] = float((~say[pres]).mean())
        lat = e.get("latency") or {}
        row.update(lat_1q_ms=lat.get("1q", {}).get("ms_median"), lat_8q_ms=lat.get("8q", {}).get("ms_median"),
                   steps_1q=lat.get("1q", {}).get("decode_steps_mean"))
        tab.append(row)
    json.dump(tab, open(f"{ROOT}/main_table.json", "w"), indent=1)
    keys = list(dict.fromkeys(k for t in tab for k in t))
    with open(f"{ROOT}/main_table.md", "w") as f:
        f.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
        for t in tab:
            f.write("| " + " | ".join(_fmt(t.get(k)) for k in keys) + " |\n")
    print(open(f"{ROOT}/main_table.md").read())

    # ------------------------------------------------------------------ surface-form competition diagnostics
    diag = {}
    for r, rows in R.items():
        if rows is None or E[r]["variant"] == "ours":
            continue
        P = present_rows(rows)
        A = [x for x in rows["grefcoco_val_merged"] if not x["present"]]
        gn = np.array([x["greedy_none"] for x in P]); pn = np.array([x["p_none"] for x in P])
        pm = np.array([x["p_maxcoord"] for x in P])
        gnA = np.array([x["greedy_none"] for x in A]); pnA = np.array([x["p_none"] for x in A])
        diag[r] = dict(
            n_present=len(P), refusal_greedy=float(gn.mean()), refusal_thresh=float((pn >= .5).mean()),
            sfc_refusals=float((gn & (pn < .5)).mean()),          # refused although P(box) > 0.5
            sfc_share_of_refusals=float((gn & (pn < .5)).sum() / max(1, gn.sum())),
            mean_pmaxcoord_present=float(pm.mean()), median_pnone_present=float(np.median(pn)),
            halluc_greedy=float(1 - gnA.mean()), halluc_thresh=float((pnA < .5).mean()),
            absent_halluc_but_pnone_gt_half=float(((~gnA) & (pnA >= .5)).mean()),
            mean_pnone_absent=float(pnA.mean()))
    json.dump(diag, open(f"{ROOT}/sfc_diagnostics.json", "w"), indent=1)
    print(json.dumps(diag, indent=1))

    # ------------------------------------------------------------------ plot 1: refusal vs entropy (present targets)
    main_runs = [r for r in ["ours_neg25", "ar_digits_neg25", "ar_hier_neg25", "ar_atomic1000_neg25",
                             "ar_atomic100_neg25"] if r in R and R[r]]
    if main_runs:
        nb = 8
        out = {}
        Pref = present_rows(R["ours_neg25"]) if "ours_neg25" in R else None
        hshared = np.array([x["H_center"] for x in Pref]) if Pref else None
        fig, axes = plt.subplots(2, len(main_runs), figsize=(3.2 * len(main_runs), 6.2), sharey=True, squeeze=False)
        STY = {"gate": ("-", "o"), "greedy": ("-", "o"), "greedy (valid tokens)": ("-.", "s"), "P(none)>=0.5": ("--", "^")}
        for k, r in enumerate(main_runs):
            P = present_rows(R[r])
            if E[r]["variant"] == "ours":
                hown = np.array([x["H_center"] + x["H_edges"] for x in P])
                rules = {"gate": np.array([x["p_present"] < .5 for x in P])}
            else:
                hown = np.array([x["H_first"] for x in P])
                rules = {"greedy": np.array([x["greedy_none"] for x in P])}
                if "greedy_none_c" in P[0]:
                    rules["greedy (valid tokens)"] = np.array([x["greedy_none_c"] for x in P])
                rules["P(none)>=0.5"] = np.array([x["p_none"] >= .5 for x in P])
            for row, (h, tag) in enumerate([(hown, "own"), (hshared, "shared")]):
                ax = axes[row, k]
                if h is None:
                    continue
                qs = np.quantile(h, np.linspace(0, 1, nb + 1))
                idx = np.clip(np.searchsorted(qs, h, side="right") - 1, 0, nb - 1)
                x = [(bb + .5) / nb * 100 for bb in range(nb)]
                for name, ref in rules.items():
                    y = [ref[idx == bb].mean() * 100 for bb in range(nb)]
                    out[f"{r}/{name}/{tag}"] = dict(x_pct=x, refusal_pct=y, bin_edges=qs.tolist(),
                                                   n_per_bin=[int((idx == bb).sum()) for bb in range(nb)])
                    ls, mk = STY[name]
                    ax.plot(x, y, ls, color=color(r), lw=2, marker=mk, ms=4, label=name)
                if row == 0:
                    ax.set_title(LABEL[r], loc="left", color=INK, fontsize=10)
                    ax.legend(fontsize=7, loc="upper left")
                ax.set_xticks([0, 25, 50, 75, 100])
                if k == 0:
                    ax.set_ylabel(("refusal %, binned by own entropy" if row == 0 else
                                   "refusal %, binned by shared difficulty"), fontsize=9)
                if row == 1:
                    ax.set_xlabel("entropy percentile", fontsize=9)
        fig.suptitle("Refusal of present targets (RefCOCO val+testA+testB, n=21,586) vs coordinate uncertainty.\n"
                     "Top: each model's own entropy (AR: first coordinate token; Ours: center+edges).  "
                     "Bottom: shared axis = Ours center-patch entropy.", x=0.01, ha="left", color=INK, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(f"{ROOT}/refusal_vs_entropy.png", dpi=150)
        json.dump(out, open(f"{ROOT}/refusal_vs_entropy.json", "w"), indent=1)

    # ------------------------------------------------------------------ plot 2: hallucination vs negative ratio
    pairs = [("ours", "ours_neg05", "ours_neg25", "gate"), ("ar_digits", "ar_digits_neg05", "ar_digits_neg25", "greedy"),
             ("ar_digits", "ar_digits_neg05", "ar_digits_neg25", "thresh")]
    if any(a in E and b in E for _, a, b, _ in pairs):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        neg = {}
        for fam, a, b, rule in pairs:
            neg_pc = None
            if not (a in E and b in E):
                continue
            ys = [E[x]["sets"]["grefcoco_val_merged"][f"halluc_{rule}"] * 100 for x in (a, b)]
            rs = [E[x]["sets"]["refcoco_val"][f"refusal_present_{rule}"] * 100 for x in (a, b)]
            lab = f"{LABEL[b].split(' (')[0]} ({rule})"
            ls = "--" if rule == "thresh" else "-"
            axes[0].plot([5, 25], ys, ls, color=color(b), lw=2, marker="o", ms=6, label=lab)
            axes[1].plot([5, 25], rs, ls, color=color(b), lw=2, marker="o", ms=6, label=lab)
            off = {"gate": [(8, 4), (8, 4)], "greedy": [(8, -10), (8, -10)], "thresh": [(8, -12), (-30, -12)]}[rule]
            for (x, y), o in zip(zip([5, 25], ys), off):
                axes[0].annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=o, fontsize=8, color=INK2)
            if rule in ("gate", "thresh"):   # post-hoc prior (logit) correction of the 5% model
                sc_rows = R.get(a) and R[a]["grefcoco_val_merged"]
                if sc_rows:
                    scv = np.array([x["p_present"] if fam == "ours" else 1 - x["p_none"] for x in sc_rows])
                    absn = np.array([not x["present"] for x in sc_rows])
                    lo = np.log(np.clip(scv, 1e-12, 1 - 1e-12)) - np.log(np.clip(1 - scv, 1e-12, 1))
                    hc = float((lo[absn] - np.log((0.95 / 0.05) / (0.75 / 0.25)) >= 0).mean() * 100)
                    axes[0].plot([5], [hc], marker="D", ms=7, mfc="none", mew=1.8, color=color(b), ls="none",
                                 label=f"{lab}, 5% + prior correction")
                    axes[0].annotate(f"{hc:.1f}", (5, hc), textcoords="offset points", xytext=(8, -4), fontsize=8, color=INK2)
                    neg[f"{fam}/{rule}"] = None  # placeholder, filled below
                    neg_pc = hc
            neg[f"{fam}/{rule}"] = dict(neg_ratio=[5, 25], halluc_pct=ys, refusal_present_pct=rs,
                                        halluc_pct_5pct_prior_corrected=neg_pc if rule in ("gate", "thresh") else None)
        axes[0].set_title("hallucination on gRefCOCO no-target (%)", loc="left", color=INK)
        axes[1].set_title("refusal on RefCOCO val present (%)", loc="left", color=INK)
        for ax in axes:
            ax.set_xlabel("absent ratio in training (%)"); ax.set_xticks([5, 25])
        axes[0].legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.12, 0.0))
        fig.tight_layout()
        fig.savefig(f"{ROOT}/halluc_vs_negratio.png", dpi=150)
        json.dump(neg, open(f"{ROOT}/halluc_vs_negratio.json", "w"), indent=1)

    # ------------------------------------------------------------------ plot 3: quantization (atomic 100 vs 1000)
    qr = [r for r in ["ar_atomic100_neg25", "ar_atomic1000_neg25"] if r in E]
    if len(qr) == 2:
        mets = [("refusal_present_greedy", "refcoco_val", "refusal present\n(greedy)"),
                ("refusal_present_thresh", "refcoco_val", "refusal present\n(P(none)≥.5)"),
                ("halluc_greedy", "grefcoco_val_merged", "halluc absent\n(greedy)"),
                ("halluc_thresh", "grefcoco_val_merged", "halluc absent\n(P(none)<.5)"),
                ("refusal_present_seqmap", "refcoco_val", "refusal present\n(seq-MAP ~ beam)"),
                ("acc50", "refcoco_val", "RefCOCO val Acc@0.5\n(constrained decoding)")]
        fig, ax = plt.subplots(figsize=(10.5, 3.8))
        w = 0.38
        for k, r in enumerate(qr):
            SS = E[r].get("sets_c", E[r]["sets"])
            ys = [(SS if m in ("acc50", "refusal_present_seqmap") else E[r]["sets"])[s][m] * 100 for m, s, _ in mets]
            xs = np.arange(len(mets)) + (k - .5) * w
            ax.bar(xs, ys, w - 0.03, color=color(r), label=LABEL[r])
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.8, f"{y:.1f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(np.arange(len(mets))); ax.set_xticklabels([l for _, _, l in mets], fontsize=8.5)
        ax.set_ylabel("%"); ax.legend(fontsize=8); ax.grid(axis="x", visible=False)
        ax.set_title("AR-atomic: coordinate quantization 100 vs 1000 bins", loc="left", color=INK)
        fig.tight_layout()
        fig.savefig(f"{ROOT}/atomic_quantization.png", dpi=150)


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v*100:.1f}" if abs(v) <= 1.0 else f"{v:.1f}"
    return str(v)


if __name__ == "__main__":
    main()
