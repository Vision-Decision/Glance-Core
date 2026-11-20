"""Sprint metrics: re-analysis of the controlled ablation with CIs, paired tests and alternative protocols.

    python outputs/sprint_metrics/scripts/ablation_stats.py      (repo root; numpy, scipy)

Reads outputs/ablation/<run>/rows{,_c}.json (read-only), the same rows the ablation tables use
(gate: rows.json; AR: grammar-constrained rows_c.json). Writes outputs/sprint_metrics/ablation_stats.json.
All bootstraps: 2,000 resamples, seed 0, items resampled jointly across heads (rows are aligned by index).
"""
import json, sys
import numpy as np
from scipy.stats import rankdata, binomtest, norm

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
O = f"{ROOT}/outputs/ablation"
B = 2000
RUNS = ["ours_neg25", "ar_digits_neg25", "ar_hier_neg25", "ar_atomic1000_neg25", "ar_atomic100_neg25",
        "ours_neg05", "ar_digits_neg05"]
AR25 = ["ar_digits_neg25", "ar_hier_neg25", "ar_atomic1000_neg25", "ar_atomic100_neg25"]


def load(r):
    f = "rows.json" if r.startswith("ours") else "rows_c.json"
    return json.load(open(f"{O}/{r}/{f}"))


R = {r: load(r) for r in RUNS}
isgate = lambda r: r.startswith("ours")


def pres_score(r, x):
    return x["p_present"] if isgate(r) else 1 - x["p_none"]


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    rk = rankdata(p); npos, nneg = (y == 1).sum(), (y == 0).sum()
    return float((rk[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ece(p, y, n_bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def rc_risk(score, corr):
    o = np.argsort(-score, kind="stable"); c = corr[o]
    return 1 - np.cumsum(c) / np.arange(1, len(c) + 1)


def aurc(score, corr):
    return float(rc_risk(score, corr).mean())


def risk_at(score, corr, cov):
    k = max(1, int(round(cov * len(corr)))); return float(rc_risk(score, corr)[k - 1])


def oracle_aurc(corr):
    return aurc(corr.astype(float), corr)


def ci(v):
    v = np.asarray(v); return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def pboot(v):  # two-sided bootstrap p-value for H0: diff = 0
    v = np.asarray(v); p = 2 * min((v <= 0).mean(), (v >= 0).mean()); return float(min(1.0, max(p, 1 / len(v))))


rng = np.random.default_rng(0)
out = {}

# ---------------------------------------------------------------- gRefCOCO val merged, per head arrays
G = {}
for r in RUNS:
    gm = R[r]["grefcoco_val_merged"]
    G[r] = dict(
        s=np.array([pres_score(r, x) for x in gm]),
        present=np.array([bool(x["present"]) for x in gm]),
        single=np.array([bool(x["present"]) and "iou" in x for x in gm]),
        multi=np.array([bool(x["present"]) and "iou" not in x for x in gm]),
        absent=np.array([not x["present"] for x in gm]),
        iou=np.array([x.get("iou", np.nan) if x["present"] else np.nan for x in gm], float),
        kappa=np.array([x.get("p_iou50", np.nan) for x in gm], float),
        lpbox=np.array([x.get("lp_box", np.nan) for x in gm], float),
        lpnone=np.array([x.get("lp_none", np.nan) for x in gm], float),
        seqmap=np.array([bool(x.get("seqmap_none", False)) for x in gm]),
    )
# alignment check (same items, same order)
for r in RUNS:
    assert (G[r]["present"] == G["ours_neg25"]["present"]).all() and (G[r]["single"] == G["ours_neg25"]["single"]).all()
N = len(G["ours_neg25"]["s"])
single, absent, multi = G["ours_neg25"]["single"], G["ours_neg25"]["absent"], G["ours_neg25"]["multi"]
present = G["ours_neg25"]["present"]
grefonly = ~single  # gRefCOCO-only = drop RefCOCO single-target positives
iS, iA, iM = np.where(single)[0], np.where(absent)[0], np.where(multi)[0]

# stratified bootstrap index sets (singles, absents, multis resampled separately; shared across heads = paired)
BOOT = [(rng.choice(iS, len(iS)), rng.choice(iA, len(iA)), rng.choice(iM, len(iM))) for _ in range(B)]

# ---------------------------------------------------------------- 1. presence AUROC/ECE with CIs, paired diffs
sec = {}
for r in RUNS:
    s, y = G[r]["s"], present.astype(int)
    a_full = auroc(s, y); a_g = auroc(s[grefonly], y[grefonly]); e_full = ece(s, y); e_g = ece(s[grefonly], y[grefonly])
    sec[r] = dict(auroc_merged=a_full, auroc_grefonly=a_g, ece_merged=e_full, ece_grefonly=e_g,
                  # absence-vs-single-target only (the matched-refusal population)
                  auroc_single_vs_absent=auroc(s[single | absent], y[single | absent]))
# bootstrap AUROC diff vs gate on single+multi+absent (merged) -- 500 resamples (AUROC is O(n log n))
bA = {r: [] for r in RUNS if not r.startswith("ours")}
bAabs = {r: [] for r in bA}
for bi in range(500):
    bs, ba, bm = BOOT[bi]; idx = np.concatenate([bs, ba, bm]); y = present[idx].astype(int)
    g0 = {g_: auroc(G[g_]["s"][idx], y) for g_ in ("ours_neg25", "ours_neg05")}
    for r in bA:
        bA[r].append(auroc(G[r]["s"][idx], y) - g0["ours_neg05" if r.endswith("neg05") else "ours_neg25"])
for r in bA:
    sec[r]["auroc_merged_minus_gate"] = [float(np.mean(bA[r])), *ci(bA[r]), pboot(bA[r])]
# CI on gate AUROC itself
bg = []
for bi in range(500):
    bs, ba, bm = BOOT[bi]; idx = np.concatenate([bs, ba, bm]); bg.append(auroc(G["ours_neg25"]["s"][idx], present[idx].astype(int)))
sec["ours_neg25"]["auroc_merged_ci"] = ci(bg)
out["presence_detection"] = sec

# ---------------------------------------------------------------- 2. matched operating points (both directions)
def nacc_at_refusal(s, sidx, aidx, fr):
    thr = np.quantile(s[sidx], fr); return float((s[aidx] < thr).mean())


def refusal_at_nacc(s, sidx, aidx, na):  # threshold so that N-acc = na; return false refusal of singles
    thr = np.quantile(s[aidx], na); return float((s[sidx] < thr).mean())


mp = {}
for r in RUNS:
    s = G[r]["s"]
    d = {f"nacc_at_{fr*100:g}pct_refusal": nacc_at_refusal(s, iS, iA, fr) for fr in (0.005, 0.01, 0.02, 0.05)}
    d.update({f"refusal_at_nacc_{int(na*100)}": refusal_at_nacc(s, iS, iA, na) for na in (0.75, 0.80, 0.85)})
    # also: TPR at FPR style on absent-detection = same as nacc_at_refusal; and default-rule 0.5 point
    d["nacc_at_0.5"] = float((s[iA] < 0.5).mean()); d["refusal_single_at_0.5"] = float((s[iS] < 0.5).mean())
    d["refusal_multi_at_0.5"] = float((s[iM] < 0.5).mean())
    mp[r] = d
# paired bootstrap of every matched metric, AR minus gate, 25% runs, and 5%: digits minus gate
pairs = [(r, "ours_neg25") for r in AR25] + [("ar_digits_neg05", "ours_neg05")]
for (r, g) in pairs:
    acc = {k: [] for k in ("nacc_at_1pct_refusal", "nacc_at_2pct_refusal", "refusal_at_nacc_80", "nacc_at_0.5")}
    for bs, ba, bm in BOOT[:1000]:
        for k in acc:
            if k.startswith("nacc_at_") and "pct" in k:
                fr = float(k.split("_")[2].replace("pct", "")) / 100
                f = lambda rr: nacc_at_refusal(G[rr]["s"], bs, ba, fr)
            elif k == "refusal_at_nacc_80":
                f = lambda rr: refusal_at_nacc(G[rr]["s"], bs, ba, .80)
            else:
                f = lambda rr: float((G[rr]["s"][ba] < .5).mean())
            acc[k].append(f(r) - f(g))
    mp[r]["paired_minus_gate"] = {k: [float(np.mean(v)), *ci(v), pboot(v)] for k, v in acc.items()}
out["matched_operating_points"] = mp

# ---------------------------------------------------------------- 3. selective prediction: AURC, E-AURC, gate vs MAP
sel_idx = np.where(single | absent)[0]
corr_all = {r: np.where(single, G[r]["iou"] >= 0.5, False) for r in RUNS}
sel = {}
for r in RUNS:
    c = corr_all[r][sel_idx]; s = G[r]["s"][sel_idx]
    d = dict(n=int(len(sel_idx)), acc_base=float(c.mean()), aurc=aurc(s, c), aurc_oracle=oracle_aurc(c))
    d["e_aurc"] = d["aurc"] - d["aurc_oracle"]
    d["risk_at_cov50"] = risk_at(s, c, .5)
    d["selective_auroc"] = auroc(s, c.astype(int))   # AUROC of score for 'answer is correct' (threshold-free)
    if isgate(r):
        gk = (G[r]["s"] * G[r]["kappa"])[sel_idx]
        d.update(aurc_gk=aurc(gk, c), e_aurc_gk=aurc(gk, c) - d["aurc_oracle"])
    else:
        mg = (G[r]["lpbox"] - G[r]["lpnone"])[sel_idx]
        d.update(aurc_margin=aurc(mg, c), e_aurc_margin=aurc(mg, c) - d["aurc_oracle"],
                 selective_auroc_margin=auroc(mg, c.astype(int)))
        acc_ = ~G[r]["seqmap"][sel_idx]
        d["map_cov"] = float(acc_.mean()); d["map_risk"] = float(1 - c[acc_].mean())
    sel[r] = d
# paired bootstrap: gate risk at each MAP coverage minus MAP risk; AURC diffs; risk@50 diffs
selS = np.where(single[sel_idx])[0]; selA = np.where(absent[sel_idx])[0]
SB = [(rng.choice(selS, len(selS)), rng.choice(selA, len(selA))) for _ in range(1000)]
g = "ours_neg25"
for r in AR25:
    acc = {k: [] for k in ("gate_minus_map_risk_at_map_cov", "gate_minus_arthr_risk_at_map_cov", "gate_minus_ar_aurc",
                           "gate_minus_ar_margin_aurc", "gate_minus_ar_risk50", "gate_minus_ar_eaurc")}
    cg_all, sg_all = corr_all[g][sel_idx], G[g]["s"][sel_idx]
    cr_all, sr_all = corr_all[r][sel_idx], G[r]["s"][sel_idx]
    mr_all = (G[r]["lpbox"] - G[r]["lpnone"])[sel_idx]; acc_all = ~G[r]["seqmap"][sel_idx]
    for bs, ba in SB:
        idx = np.concatenate([bs, ba])
        cg, sg, cr, sr, mr, am = cg_all[idx], sg_all[idx], cr_all[idx], sr_all[idx], mr_all[idx], acc_all[idx]
        cov = am.mean(); maprisk = 1 - cr[am].mean()
        acc["gate_minus_map_risk_at_map_cov"].append(risk_at(sg, cg, cov) - maprisk)
        acc["gate_minus_arthr_risk_at_map_cov"].append(risk_at(sg, cg, cov) - risk_at(sr, cr, cov))
        ag, ar_ = aurc(sg, cg), aurc(sr, cr)
        acc["gate_minus_ar_aurc"].append(ag - ar_)
        acc["gate_minus_ar_margin_aurc"].append(ag - aurc(mr, cr))
        acc["gate_minus_ar_risk50"].append(risk_at(sg, cg, .5) - risk_at(sr, cr, .5))
        acc["gate_minus_ar_eaurc"].append((ag - oracle_aurc(cg)) - (ar_ - oracle_aurc(cr)))
    sel[r]["paired_vs_gate"] = {k: [float(np.mean(v)), *ci(v), pboot(v)] for k, v in acc.items()}
out["selective"] = sel

# ---------------------------------------------------------------- 4. box accuracy: McNemar gate vs AR per split, Acc@.75/.9
box = {}
splits = {"refcoco_val": None, "refcoco_testA": "refcoco_testA", "refcoco_testB": "refcoco_testB"}


def split_iou(r, sp):
    if sp == "refcoco_val":
        return G[r]["iou"][single]
    return np.array([x["iou"] for x in R[r][sp] if x["present"] and "iou" in x], float)


for sp in splits:
    ig = split_iou("ours_neg25", sp)
    for r in RUNS:
        iu = split_iou(r, sp)
        box.setdefault(r, {})[sp] = {f"acc{t}": float((iu >= t / 100).mean()) for t in (50, 75, 90)}
        box[r][sp]["miou"] = float(iu.mean())
        if r != "ours_neg25" and r.endswith("neg25"):
            a, b_ = ig >= .5, iu >= .5
            n01, n10 = int((a & ~b_).sum()), int((~a & b_).sum())
            p = binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0
            box[r][sp]["mcnemar_gate_wins_ar_wins_p"] = [n01, n10, float(p)]
    if True:
        ig5 = split_iou("ours_neg05", sp); iu5 = split_iou("ar_digits_neg05", sp)
        a, b_ = ig5 >= .5, iu5 >= .5; n01, n10 = int((a & ~b_).sum()), int((~a & b_).sum())
        box["ar_digits_neg05"][sp]["mcnemar_gate_wins_ar_wins_p"] = [n01, n10, float(binomtest(n01, n01 + n10, .5).pvalue)]
# pooled over the three splits (21,586)
for r in RUNS:
    iu = np.concatenate([split_iou(r, sp) for sp in splits])
    box[r]["pooled3"] = {"n": int(len(iu)), "acc50": float((iu >= .5).mean()), "acc75": float((iu >= .75).mean()),
                         "acc90": float((iu >= .9).mean()), "miou": float(iu.mean()),
                         # COCO-style mean Acc over IoU .5:.95
                         "acc50_95": float(np.mean([(iu >= t).mean() for t in np.arange(.5, .96, .05)]))}
ig = np.concatenate([split_iou("ours_neg25", sp) for sp in splits])
for r in AR25:
    iu = np.concatenate([split_iou(r, sp) for sp in splits])
    a, b_ = ig >= .5, iu >= .5; n01, n10 = int((a & ~b_).sum()), int((~a & b_).sum())
    box[r]["pooled3"]["mcnemar_gate_wins_ar_wins_p"] = [n01, n10, float(binomtest(n01, n01 + n10, .5).pvalue)]
    d = []
    for _ in range(1000):
        i = rng.integers(0, len(ig), len(ig)); d.append(iu[i].mean() - ig[i].mean())
    box[r]["pooled3"]["miou_minus_gate"] = [float(np.mean(d)), *ci(d)]
out["box"] = box

# ---------------------------------------------------------------- 5. decision-rule refusal: greedy/threshold ratio CIs, MAP CIs
dr = {}
for r in AR25 + ["ar_digits_neg05"]:
    Pr = []
    for s_ in ("grefcoco_val_merged", "refcoco_testA", "refcoco_testB"):
        Pr += [x for x in R[r][s_] if x["present"] and "iou" in x]
    gn = np.array([x["greedy_none"] for x in Pr]); th = np.array([x["p_none"] >= .5 for x in Pr])
    sm = np.array([x["seqmap_none"] for x in Pr])
    rat = []
    for _ in range(2000):
        i = rng.integers(0, len(Pr), len(Pr)); rat.append(gn[i].mean() / max(th[i].mean(), 1e-9))
    dr[r] = dict(n=len(Pr), greedy=float(gn.mean()), thresh=float(th.mean()), seqmap=float(sm.mean()),
                 ratio=float(gn.mean() / th.mean()), ratio_ci=ci(rat), n_greedy=int(gn.sum()), n_thresh=int(th.sum()),
                 greedy_only=int((gn & ~th).sum()), thresh_only=int((th & ~gn).sum()),
                 mcnemar_p=float(binomtest(int((gn & ~th).sum()), int((gn ^ th).sum()), .5).pvalue))
out["decision_rules"] = dr

# ---------------------------------------------------------------- 6. negative-ratio effect with CIs (hallucination at 0.5)
nr = {}
for a5, a25 in (("ours_neg05", "ours_neg25"), ("ar_digits_neg05", "ar_digits_neg25")):
    h = lambda r, ba: 1 - float((G[r]["s"][ba] < .5).mean())
    d = [h(a5, ba) - h(a25, ba) for _, ba, _ in BOOT[:1000]]
    nr[a5] = dict(halluc_25=h(a25, iA), halluc_05=h(a5, iA), diff=[float(np.mean(d)), *ci(d)])
out["neg_ratio"] = nr

# ---------------------------------------------------------------- 7. run-to-run proxy: v0 vs gate run (same recipe)
v0 = json.load(open(f"{ROOT}/outputs/v0/eval_results.json"))["sets"]["grefcoco_val_merged"]
out["run_to_run_proxy"] = dict(v0_N_acc=v0["N_acc"], gate_N_acc=mp["ours_neg25"]["nacc_at_0.5"],
                               v0_auroc=v0["auroc_presence"], gate_auroc=sec["ours_neg25"]["auroc_merged"],
                               v0_ece=v0["ece_presence"], gate_ece=sec["ours_neg25"]["ece_merged"])

# ---------------------------------------------------------------- 8. positives only: IoU head vs AR box scores
po = {}
def risk_cov(score, err, cov):
    o = np.argsort(-score, kind="stable"); k = int(round(cov * len(err))); return float(err[o][:k].mean())
for r in ["ours_neg25"] + AR25:
    err = ~(G[r]["iou"][single] >= .5)
    sc = {"kappa": G[r]["kappa"][single]} if isgate(r) else {"lp_box": G[r]["lpbox"][single], "margin": (G[r]["lpbox"] - G[r]["lpnone"])[single], "1-p_none": G[r]["s"][single]}
    if isgate(r): sc["gate"] = G[r]["s"][single]
    d = {"base_error": float(err.mean())}
    for k, v in sc.items():
        bs_ = []
        for _ in range(500):
            i = rng.integers(0, len(v), len(v)); bs_.append(auroc(-v[i], err[i].astype(int)))
        d[k] = dict(auroc_err=auroc(-v, err.astype(int)), auroc_ci=ci(bs_), risk_at_cov80=risk_cov(v, err, .8),
                    risk_at_cov90=risk_cov(v, err, .9), aurc=aurc(v, ~err))
    po[r] = d
out["positives_only"] = po

json.dump(out, open(f"{ROOT}/outputs/sprint_metrics/ablation_stats.json", "w"), indent=1)
print("ok")
