"""Sprint metrics: v0/27B/v1 main results, GUI and OpenDecisions under alternative protocols, with CIs.

    python outputs/sprint_metrics/scripts/main_stats.py [repo_root] [opendecisions_dir]

opendecisions_dir holds the benchmark's score.py and opendecisions.jsonl (HF jgeuter/OpenDecisions); optional.
Writes outputs/sprint_metrics/main_stats.json. Read-only on outputs/.
"""
import json, sys, importlib.util
import numpy as np
from scipy.stats import rankdata, norm, beta

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
ODDIR = sys.argv[2] if len(sys.argv) > 2 else None
O = f"{ROOT}/outputs"
rng = np.random.default_rng(0)
J = lambda p: json.load(open(f"{O}/{p}"))
out = {}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def ci(v):
    v = np.asarray(v); return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int); rk = rankdata(p); a, b = (y == 1).sum(), (y == 0).sum()
    return float((rk[y == 1].sum() - a * (a + 1) / 2) / (a * b))


def ece(p, y, n_bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean()) for b in range(n_bins) if (bins == b).any()))


SPLITS8 = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocoplus_testA",
           "refcocoplus_testB", "refcocog_val", "refcocog_test"]

# ------------------------------------------------------------------ 1. RefCOCO family: averaging protocols + CIs
ref = {}
for name, path in (("v0_4b", "v0/eval_results.json"), ("v0_27b", "27b/eval_results.json"), ("v1_4b", "v1/eval_results.json")):
    s = J(path)["sets"]
    d = {k: dict(acc50=s[k]["acc50"], n=s[k]["n"], ci95=wilson(round(s[k]["acc50"] * s[k]["n"]), s[k]["n"]),
                 gated=s[k].get("gated_acc50"), miou=s[k].get("mean_iou")) for k in SPLITS8}
    accs = np.array([s[k]["acc50"] for k in SPLITS8]); ns = np.array([s[k]["n"] for k in SPLITS8])
    d["avg8_macro"] = float(accs.mean()); d["avg8_micro"] = float((accs * ns).sum() / ns.sum())
    d["avg8_gated_macro"] = float(np.mean([s[k]["gated_acc50"] for k in SPLITS8]))
    test = ["refcoco_testA", "refcoco_testB", "refcocoplus_testA", "refcocoplus_testB", "refcocog_test"]
    d["avg5_test_macro"] = float(np.mean([s[k]["acc50"] for k in test]))   # Jag-style (test splits only)
    d["avg_refcoco3"] = float(np.mean([s[k]["acc50"] for k in SPLITS8[:3]]))
    d["avg_refcocoplus3"] = float(np.mean([s[k]["acc50"] for k in SPLITS8[3:6]]))
    d["avg_refcocog2"] = float(np.mean([s[k]["acc50"] for k in SPLITS8[6:]]))
    # macro over the three datasets (each dataset = mean of its splits) -- another common 'avg'
    d["avg_by_dataset"] = float(np.mean([d["avg_refcoco3"], d["avg_refcocoplus3"], d["avg_refcocog2"]]))
    # SE of the 8-split mean (independent splits; binomial)
    d["avg8_se"] = float(np.sqrt(sum(a * (1 - a) / n for a, n in zip(accs, ns))) / 8)
    g = s["grefcoco_val_merged"]; gc = g.get("calibrated", g)
    nA, nM, nS = 8905, 5324, 10834
    d["absence"] = dict(N_acc=g["N_acc"], N_acc_ci=wilson(round(g["N_acc"] * nA), nA),
                        T_acc_single=g["T_acc_single"], recall_multi=g["present_recall_multi"],
                        refusal_multi_ci=wilson(round((1 - g["present_recall_multi"]) * nM), nM),
                        auroc=g["auroc_presence"], ece_raw=g["ece_presence"], ece_cal=gc["ece_presence"],
                        # balanced accuracy-style summaries at the 0.5 gate (threshold rule, single + multi present)
                        bal_acc_absent_vs_single=None)
    ref[name] = d
# Jag-0.8B / LOCUS-2B comparisons on overlapping splits
jag = dict(refcoco_testA=93.9, refcoco_testB=88.4, refcocoplus_testA=90.7, refcocoplus_testB=80.1, refcocog_test=87.7)
locus = dict(zip(SPLITS8, [93.1, 94.3, 90.7, 87.9, 91.9, 83.6, 88.7, 89.9]))
ref["vs_jag_test5"] = {m: float(np.mean([ref[m][k]["acc50"] * 100 for k in jag])) for m in ("v0_4b", "v0_27b", "v1_4b")}
ref["vs_jag_test5"]["jag_0.8b"] = float(np.mean(list(jag.values())))
ref["vs_locus_avg8"] = {m: ref[m]["avg8_macro"] * 100 for m in ("v0_4b", "v0_27b", "v1_4b")}
ref["vs_locus_avg8"]["locus_2b"] = float(np.mean(list(locus.values())))
ref["per_split_wins_v1_vs_locus"] = sum(ref["v1_4b"][k]["acc50"] * 100 > locus[k] for k in SPLITS8)
out["refcoco"] = ref

# ------------------------------------------------------------------ 2. v0 per-row: absence metrics, CIs, GREC-style
rows = J("v0/eval_rows_refcocoval_grefcocoval.json")
rv, gv = rows["refcoco_val"], rows["grefcoco_val"]
ps = np.array([x["p_present"] for x in rv]); iou = np.array([x["iou"] for x in rv])
pg = np.array([x["p_present"] for x in gv]); yg = np.array([x["present"] for x in gv])
pa, pm = pg[~yg], pg[yg]
v0 = dict(n_single=len(ps), n_absent=int((~yg).sum()), n_multi=int(yg.sum()))
v0["N_acc"] = float((pa < .5).mean()); v0["N_acc_ci"] = wilson(int((pa < .5).sum()), len(pa))
v0["false_refusal_single"] = float((ps < .5).mean()); v0["refusal_multi"] = float((pm < .5).mean())
v0["T_acc_single_gated"] = float(((ps >= .5) & (iou >= .5)).mean())
allp = np.concatenate([ps, pg]); ally = np.concatenate([np.ones(len(ps), bool), yg])
v0["auroc_merged"] = auroc(allp, ally); v0["auroc_grefonly"] = auroc(pg, yg)
v0["auroc_single_vs_absent"] = auroc(np.concatenate([ps, pa]), np.concatenate([np.ones(len(ps)), np.zeros(len(pa))]))
v0["ece_merged"] = ece(allp, ally); v0["ece_grefonly"] = ece(pg, yg)
for fr in (.005, .01, .02):
    v0[f"N_acc_at_{fr*100:g}pct_refusal"] = float((pa < np.quantile(ps, fr)).mean())
# operating point maximizing balanced accuracy (absent vs single), reported for transparency only
ths = np.linspace(.01, .99, 99)
ba_ = [((pa < t).mean() + (ps >= t).mean()) / 2 for t in ths]
v0["best_balacc_thr_on_eval"] = [float(ths[int(np.argmax(ba_))]), float(max(ba_))]
bt = []
for _ in range(1000):
    i, j, k = rng.integers(0, len(ps), len(ps)), rng.integers(0, len(pa), len(pa)), rng.integers(0, len(pm), len(pm))
    p_ = np.concatenate([ps[i], pa[j], pm[k]]); y_ = np.concatenate([np.ones(len(i)), np.zeros(len(j)), np.ones(len(k))])
    bt.append((auroc(p_, y_), ece(p_, y_)))
bt = np.array(bt); v0["auroc_merged_ci"] = ci(bt[:, 0]); v0["ece_merged_ci"] = ci(bt[:, 1])
out["v0_rows"] = v0

# ------------------------------------------------------------------ 3. GUI: micro vs macro, CIs, Basic/Advanced
v1 = J("v1/eval_results.json")["sets"]
gui = {}
for k in ("ss2", "sspro", "venus"):
    r = v1[k]["raw"]; bg = r["by_group"]
    d = dict(n=r["n"], click_micro=r["click_acc"], click_ci=None)
    if k != "venus":
        d["click_ci"] = wilson(round(r["click_acc"] * r["n"]), r["n"])
        d["click_macro_groups"] = float(np.mean([v["click_acc"] for v in bg.values()]))
        for t in ("text", "icon"):
            gs = [v for g_, v in bg.items() if g_.endswith(t)]
            d[f"{t}_micro"] = sum(v["click_acc"] * v["n"] for v in gs) / sum(v["n"] for v in gs)
        plats = sorted({g_.split("/")[0] for g_ in bg})
        d["by_platform"] = {p: sum(bg[f"{p}/{t}"]["click_acc"] * bg[f"{p}/{t}"]["n"] for t in ("text", "icon")) /
                            sum(bg[f"{p}/{t}"]["n"] for t in ("text", "icon")) for p in plats}
    else:
        npos = sum(v["n"] for v in bg.values()); nref = r["n"] - npos
        d.update(n_pos=npos, n_ref=nref, pos_click=r["click_acc"], refusal=r["refusal_acc"], auroc=r["auroc_presence"],
                 pos_ci=wilson(round(r["click_acc"] * npos), npos), ref_ci=wilson(round(r["refusal_acc"] * nref), nref))
        d["overall_micro"] = (r["click_acc"] * npos + r["refusal_acc"] * nref) / r["n"]
        d["overall_ci"] = wilson(round(d["overall_micro"] * r["n"]), r["n"])
        cat = {}
        for g_, v in bg.items():
            c = g_.split("/")[1]; cat.setdefault(c, [0, 0]); cat[c][0] += v["click_acc"] * v["n"]; cat[c][1] += v["n"]
        cat = {c: a / n for c, (a, n) in cat.items()}; cat["refusal_grounding"] = r["refusal_acc"]
        d["by_category"] = cat
        catn = {}
        for g_, v in bg.items():
            catn[g_.split("/")[1]] = catn.get(g_.split("/")[1], 0) + v["n"]
        catn["refusal_grounding"] = nref; d["category_n"] = catn
        d["macro_over_6_categories"] = float(np.mean(list(cat.values())))
        basic = ["element_grounding", "spatial_grounding", "visual_grounding"]
        adv = ["functional_grounding", "reason_grounding", "refusal_grounding"]
        d["basic_micro_ASSUMED_SPLIT"] = sum(cat[c] * catn[c] for c in basic) / sum(catn[c] for c in basic)
        d["advanced_micro_ASSUMED_SPLIT"] = sum(cat[c] * catn[c] for c in adv) / sum(catn[c] for c in adv)
        d["advanced_without_refusal"] = sum(cat[c] * catn[c] for c in adv[:2]) / sum(catn[c] for c in adv[:2])
        plats = {}
        for g_, v in bg.items():
            p = g_.split("/")[0]; plats.setdefault(p, [0, 0]); plats[p][0] += v["click_acc"] * v["n"]; plats[p][1] += v["n"]
        d["pos_by_platform"] = {p: a / n for p, (a, n) in plats.items()}
    gui[k] = d
# cross-domain rows (v0 and base)
xd = {}
for f in ("base_ss2_4000", "base_sspro_4000", "ours_ss2_640"):
    rr = [json.loads(l) for l in open(f"{O}/eval_v0_xdomain/rows/{f}.jsonl")]
    h = np.array([bool(x["hit"]) for x in rr]); d = dict(n=len(rr), acc=float(h.mean()), ci=wilson(int(h.sum()), len(h)))
    grp = {}
    for x in rr:
        m = x.get("meta", {}); key = f"{m.get('platform', m.get('group', '?'))}/{m.get('ui_type', '?')}"
        grp.setdefault(key, []).append(bool(x["hit"]))
    d["by_group"] = {k_: [float(np.mean(v)), len(v)] for k_, v in sorted(grp.items())}
    d["macro_groups"] = float(np.mean([np.mean(v) for v in grp.values()]))
    if "ms" in rr[0]:
        d["ms_median"] = float(np.median([x["ms"] for x in rr]))
    if "n_in_tokens" in rr[0]:
        d["in_tokens_median"] = float(np.median([x["n_in_tokens"] for x in rr]))
    xd[f] = d
gui["xdomain"] = xd
out["gui"] = gui

# ------------------------------------------------------------------ 4. OpenDecisions: paired bootstrap with the official scorer
if ODDIR:
    spec = importlib.util.spec_from_file_location("odscore", f"{ODDIR}/score.py"); S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    tasks = S.read_jsonl(f"{ODDIR}/opendecisions.jsonl")
    def preds(p):
        return {json.loads(l)["id"]: json.loads(l)["probs"] for l in open(f"{O}/{p}")}
    P = {"v1_raw": preds("v1/eval_results_od_raw.jsonl"), "v1_cal": preds("v1/eval_results_od_cal.jsonl"),
         "base_raw": preds("v1/eval_base_qwen35_4b_od_raw.jsonl"), "base_cal": preds("v1/eval_base_qwen35_4b_od_cal.jsonl")}
    P["v1_clipped"] = {t["id"]: (P["v1_raw"] if t["question"]["type"] == "noul" else P["v1_cal"])[t["id"]] for t in tasks}
    OUT = {k: [S.item_outcome(t, S.normalized(t, v[t["id"]])) for t in tasks] for k, v in P.items()}
    od = {k: S.summary(v) for k, v in OUT.items()}
    od = {k: {m: v[m] for m in ("capability", "intelligence", "calibration", "accuracy_per_type", "competence_per_type",
                                "calibration_per_type")} for k, v in od.items()}
    # stratified (by type x tier) paired bootstrap over items
    strata = {}
    for i, t in enumerate(tasks):
        strata.setdefault((t["question"]["type"], t["tier"]), []).append(i)
    strata = [np.array(v) for v in strata.values()]
    keys = ["v1_clipped", "v1_cal", "base_cal", "v1_raw", "base_raw"]
    bs = {k: [] for k in keys}
    for _ in range(1000):
        idx = np.concatenate([rng.choice(s_, len(s_)) for s_ in strata])
        for k in keys:
            sm = S.summary([OUT[k][i] for i in idx]); bs[k].append((sm["capability"], sm["intelligence"], sm["calibration"]))
    bs = {k: np.array(v) for k, v in bs.items()}
    od["ci"] = {k: {m: ci(bs[k][:, j]) for j, m in enumerate(("capability", "intelligence", "calibration"))} for k in keys}
    od["paired"] = {}
    for a, b in (("v1_clipped", "base_cal"), ("v1_raw", "base_raw"), ("v1_cal", "base_cal")):
        dd = bs[a] - bs[b]
        od["paired"][f"{a}_minus_{b}"] = {m: [float(dd[:, j].mean()), *ci(dd[:, j]),
                                              float(min(1, 2 * min((dd[:, j] <= 0).mean(), (dd[:, j] >= 0).mean())))]
                                          for j, m in enumerate(("capability", "intelligence", "calibration"))}
    # one-sample: v1 vs the authors' 3-seed mean 66.6 (+-2.6 sd across seeds)
    od["v1_clipped_minus_66.6"] = [float(od["v1_clipped"]["capability"] - 66.6), *[x - 66.6 for x in od["ci"]["v1_clipped"]["capability"]]]
    out["opendecisions"] = od

# ------------------------------------------------------------------ 5. latency: ratios from the fair-latency run (ablation_v2)
try:
    L = json.load(open("~/Developer/decision-vison/outputs/ablation_v2/latency/latency_summary.json"))
    gk = "Glance (ours)"; vk = [k for k in L if "vLLM + prefix cache (all" in k][0]; vn = [k for k in L if "no prefix" in k][0]
    ak = [k for k in L if k.startswith("AR-digits")][0]
    lat = {}
    for N in ("bs1_N1", "bs1_N2", "bs1_N4", "bs1_N8", "bs1_N16", "bs1_N32", "bs8_N1", "bs8_N8"):
        lat[N] = dict(glance=L[gk][N]["ms_median"], vllm_pc=L[vk][N]["ms_median"], vllm_nopc=L[vn][N]["ms_median"],
                      ar_digits_kv=L[ak][N]["ms_median"],
                      x_vllm_pc=L[vk][N]["ms_median"] / L[gk][N]["ms_median"],
                      x_vllm_nopc=L[vn][N]["ms_median"] / L[gk][N]["ms_median"],
                      x_ar_kv=L[ak][N]["ms_median"] / L[gk][N]["ms_median"],
                      ttfa_x_vllm_pc=L[vk][N]["ttfa_ms_median"] / L[gk][N]["ttfa_ms_median"],
                      qps_glance=L[gk][N]["q_per_s"], qps_vllm_pc=L[vk][N]["q_per_s"])
    out["latency_fair"] = lat
except Exception as e:  # noqa
    out["latency_fair"] = str(e)

json.dump(out, open(f"{ROOT}/outputs/sprint_metrics/main_stats.json", "w"), indent=1)
print("ok")
