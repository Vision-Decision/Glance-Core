"""Score Jev runs and compare with Glance v1 on the identical items (CPU only; reads cached responses).

    python evals/jev/score.py <bench> --root ~/jev [--out summary.json]
benches: grefcoco | refcoco8 | robust | gui (ss2, sspro, venus) | attr | ambig | od | all

Jev outputs used: box `{top,bottom,left,right}` (fractions of the sent image, mapped back with the original W, H),
box `confidence`, and the parallel noul P(true) as P(present). A question answered `refus` or missing counts as a
failure: for presence it is scored as "absent" (p = 0) and for boxes as a miss (IoU 0), and counted separately.
Metric definitions follow eval.py (summarize / summarize_gui) so the Glance numbers are comparable:
  acc50 = IoU >= 0.5 ignoring presence; gated_acc50 = P >= 0.5 and IoU >= 0.5; N-acc = absent with P < 0.5;
  AUROC/ECE (15 bins) of P(present) over all items; click = box centre inside the GT box (ungated).
CIs: 95% percentile bootstrap over items, 1,000 resamples, seed 0; paired (same resample) for Jev - Glance.
"""
import argparse, collections, glob, gzip, json, math, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
SPLITS = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocoplus_testA", "refcocoplus_testB",
          "refcocog_val", "refcocog_test"]
B = 1000


def ece(p, y, n_bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean()) for b in range(n_bins) if (bins == b).any()))


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def union_box(bs):
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def boot(fn, n, paired=None, seed=0):
    """fn(idx) -> value on the resampled index set. Returns [lo, hi]."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v == v:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def mean_ci(x, paired=None):
    x = np.asarray(x, float)
    out = dict(v=float(x.mean()) if len(x) else float("nan"), n=int(len(x)))
    if len(x):
        out["ci"] = boot(lambda i: x[i].mean(), len(x))
    if paired is not None:
        y = np.asarray(paired, float)
        out["glance"] = float(y.mean()); out["diff"] = float(x.mean() - y.mean())
        out["diff_ci"] = boot(lambda i: x[i].mean() - y[i].mean(), len(x))
    return out


def auroc_ci(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    return dict(v=auroc(p, y), n=int(len(p)), ci=boot(lambda i: auroc(p[i], y[i]), len(p)))


# ------------------------------------------------------------------------------------------------ loading
def load_run(root, bench):
    """-> list of (request, raw or None)"""
    out = []
    for l in open(f"{root}/runs/{bench}/requests.jsonl"):
        r = json.loads(l)
        fn = f"{root}/runs/{bench}/raw/{r['id'].replace('/', '__')}.json"
        out.append((r, json.load(open(fn)) if os.path.exists(fn) else None))
    return out


def run_stats(run):
    lat = [raw["latency_ms"] for _, raw in run if raw]
    cost = sum(raw["resp"].get("usage", {}).get("cost", 0) or 0 for _, raw in run if raw)
    tok = sum(raw["resp"].get("usage", {}).get("input_tokens", 0) or 0 for _, raw in run if raw)
    nq = [raw["n_questions"] for _, raw in run if raw]
    return dict(requests=len(run), ok=len(lat), failed_requests=len(run) - len(lat), cost_usd=cost, input_tokens=tok,
                questions_per_request_median=float(np.median(nq)) if nq else None,
                latency_ms_median=float(np.median(lat)) if lat else None,
                latency_ms_p80=float(np.percentile(lat, 80)) if lat else None,
                note="per-request wall time under concurrent bulk load from gen-host (network + OpenRouter + Jev)")


def jev_items(run, nat=True):
    """flatten -> one dict per grounding item: p (noul), box (px), conf, flags"""
    rows, fails = [], collections.Counter()
    for req, raw in run:
        ans = (raw or {}).get("resp", {}).get("answers", {}) if raw else {}
        for it in req["items"]:
            k = it["k"]; W, H = it["W"], it["H"]
            a_box, a_yes = ans.get(f"{k}_box"), ans.get(f"{k}_yes")
            r = dict(it)
            if raw is None:
                fails["request_failed"] += 1
            if a_yes and a_yes.get("type") == "noul" and a_yes.get("noul") is not None:
                r["p"] = float(a_yes["noul"])
            else:
                r["p"] = 0.0; r["p_fail"] = (a_yes or {}).get("type", "missing"); fails[f"noul_{r['p_fail']}"] += 1
            if a_box and a_box.get("type") == "box" and a_box.get("box"):
                b = a_box["box"]
                r["pbox"] = [b["left"] * W, b["top"] * H, b["right"] * W, b["bottom"] * H]
                r["conf"] = float(a_box.get("confidence")) if a_box.get("confidence") is not None else float("nan")
            else:
                r["pbox"] = None; r["conf"] = 0.0
                r["box_fail"] = (a_box or {}).get("type", "missing"); fails[f"box_{r['box_fail']}"] += 1
            rows.append(r)
    return rows, dict(fails)


def iou_or0(pb, gb):
    return iou(pb, gb) if (pb is not None and gb is not None) else 0.0


# ------------------------------------------------------------------------------------------------ gRefCOCO
def score_grefcoco(root):
    run = load_run(root, "grefcoco")
    g = json.load(open(f"{REPO}/outputs/probes/ambiguity_attr/grefcoco_full.json"))["rows"]
    gl = {(r["img"], r["sid"]): r for r in g}  # Glance v1, calibrated P(present), full 25,063 rows
    for req, _ in run:  # image id is part of the request id
        iid = int(req["id"].split("/")[1])
        for it in req["items"]:
            it["_img"] = iid
    rows, fails = jev_items(run)
    pairs = [(r, gl.get((r["_img"], r["sent_id"]))) for r in rows]
    missing_glance = sum(g is None for _, g in pairs)
    pairs = [(r, g) for r, g in pairs if g is not None]
    jr = [r for r, _ in pairs]; gr = [g for _, g in pairs]
    pres = np.array([r["present"] for r in jr]); nt = np.array([r["n_targets"] for r in jr])
    pj = np.array([r["p"] for r in jr]); pg = np.array([g["p"] for g in gr])
    confj = np.array([r["conf"] for r in jr]); kg = np.array([g["k"] for g in gr])
    ious_j = [[iou_or0(r["pbox"], b) for b in r["boxes"]] for r in jr]
    ious_g = [g.get("iou") or [] for g in gr]
    absent = ~pres; single = pres & (nt == 1); multi = pres & (nt > 1)
    s = dict(n=len(jr), n_absent=int(absent.sum()), n_single=int(single.sum()), n_multi=int(multi.sum()),
             missing_glance_rows=missing_glance, jev_failures=fails)
    s["N_acc"] = mean_ci((pj < 0.5)[absent], (pg < 0.5)[absent])
    y = pres.astype(int)
    s["auroc_presence"] = dict(jev=auroc_ci(pj, y), glance=auroc(pg, y),
                               jev_box_confidence_as_presence=auroc(confj, y),
                               diff_ci=boot(lambda i: auroc(pj[i], y[i]) - auroc(pg[i], y[i]), len(y)))
    s["ece_presence"] = dict(jev=ece(pj, y), glance=ece(pg, y),
                             jev_ci=boot(lambda i: ece(pj[i], y[i]), len(y)))
    i50j = np.array([(max(x) if x else 0) >= 0.5 for x in ious_j]); i50g = np.array([(max(x) if x else 0) >= 0.5 for x in ious_g])
    s["acc50_single"] = mean_ci(i50j[single], i50g[single])
    s["gated_acc50_single"] = mean_ci((i50j & (pj >= 0.5))[single], (i50g & (pg >= 0.5))[single])
    s["mean_iou_single"] = dict(jev=float(np.mean([x[0] for x, m in zip(ious_j, single) if m])),
                                glance=float(np.mean([x[0] for x, m in zip(ious_g, single) if m])))
    s["false_refusal_single"] = mean_ci((pj < 0.5)[single], (pg < 0.5)[single])
    # box confidence vs IoU>=0.5 on single-target positives (Jev confidence vs Glance kappa-hat)
    ys = i50j[single].astype(int); yg = i50g[single].astype(int)
    cj, kk = confj[single], kg[single]
    s["box_conf_auroc_single"] = dict(jev=auroc_ci(cj, ys), glance=auroc_ci(kk, yg),
                                      jev_p_times_conf=auroc(pj[single] * cj, ys))
    s["box_conf_ece_single"] = dict(jev=ece(cj, ys), glance=ece(kk, yg))
    # multi-target
    hit_j = i50j[multi]; hit_g = i50g[multi]
    un_j = np.array([iou_or0(r["pbox"], union_box(r["boxes"])) >= 0.5 for r, m in zip(jr, multi) if m])
    un_g = np.array([(g.get("iou_union") or 0) >= 0.5 for g, m in zip(gr, multi) if m])
    s["multi_present_recall"] = mean_ci((pj >= 0.5)[multi], (pg >= 0.5)[multi])
    s["multi_hit_any"] = mean_ci(hit_j, hit_g)
    s["multi_union_iou50"] = mean_ci(un_j, un_g)
    # selective view: Jev N-acc at Glance-like 1% false refusal on single-target (threshold on each model's own p)
    s["N_acc_at_1pct_false_refusal"] = {}
    for name, p in (("jev", pj), ("glance", pg)):
        tau = float(np.quantile(p[single], 0.01))
        s["N_acc_at_1pct_false_refusal"][name] = dict(tau=tau, N_acc=float((p[absent] < tau).mean()))
    s["run"] = run_stats(run)
    return s


# ------------------------------------------------------------------------------------------------ RefCOCO 8 splits
def glance_rows_by_key(fn):
    out = {}
    with gzip.open(fn, "rt") as f:
        for l in f:
            r = json.loads(l)
            out[(r["text"], tuple(round(v, 1) for v in r["gt_box"]))] = r
    return out


def score_refcoco8(root):
    gres = json.load(open(f"{REPO}/outputs/v1/eval_results.json"))["sets"]
    res, accs_j, accs_g = {}, [], []
    for sp in SPLITS:
        if not os.path.exists(f"{root}/runs/{sp}/requests.jsonl"):
            continue
        run = load_run(root, sp)
        rows, fails = jev_items(run)
        hit = np.array([iou_or0(r["pbox"], r["box"]) >= 0.5 for r in rows])
        gated = np.array([(iou_or0(r["pbox"], r["box"]) >= 0.5) and r["p"] >= 0.5 for r in rows])
        gfn = f"{root}/data/glance/sprint_abs/rows_v1/{sp}.jsonl.gz"
        paired = None
        if os.path.exists(gfn):
            gk = glance_rows_by_key(gfn)
            gm = [gk.get((r["text"], tuple(round(v, 1) for v in r["box"]))) for r in rows]
            if all(x is not None for x in gm):
                paired = np.array([x["iou"] >= 0.5 for x in gm])
        s = dict(acc50=mean_ci(hit, paired), gated_acc50=float(gated.mean()),
                 glance_acc50_eval_results=gres[sp]["acc50"], glance_gated_acc50=gres[sp]["gated_acc50"],
                 jev_false_refusal=float(np.mean([r["p"] < 0.5 for r in rows])),
                 jev_failures=fails, run=run_stats(run), paired_with_glance_rows=paired is not None)
        res[sp] = s
        accs_j.append(s["acc50"]["v"]); accs_g.append(gres[sp]["acc50"])
    if len(accs_j) == 8:
        res["avg8"] = dict(jev=float(np.mean(accs_j)), glance_v1=float(np.mean(accs_g)), glance_v0=0.928)
    return res


# ------------------------------------------------------------------------------------------------ robust
def score_robust(root):
    run = load_run(root, "robust")
    rows, fails = jev_items(run)
    g = {}
    for fn in glob.glob(f"{root}/data/glance/robust_eval/v1/robust_*.jsonl.gz"):
        with gzip.open(fn, "rt") as f:
            for l in f:
                r = json.loads(l); g[r["meta"]["id"]] = r
    res = {}
    for t in ("zh", "para", "func", "vague"):
        rr = [r for r in rows if r["type"] == t]
        gg = [g.get(r["item_id"]) for r in rr]
        ok = all(x is not None for x in gg)
        pres = np.array([r["present"] for r in rr]); pj = np.array([r["p"] for r in rr])
        hit = np.array([iou_or0(r["pbox"], r["box"]) >= 0.5 if r["present"] else False for r in rr])
        s = dict(n=len(rr), n_absent=int((~pres).sum()))
        if ok:
            pg = np.array([x["p_present"] for x in gg]); hg = np.array([(x.get("iou") or 0) >= 0.5 for x in gg])
            s["acc50"] = mean_ci(hit[pres], hg[pres])
            s["gated_acc50"] = mean_ci((hit & (pj >= 0.5))[pres], (hg & (pg >= 0.5))[pres])
            s["N_acc"] = mean_ci((pj < 0.5)[~pres], (pg < 0.5)[~pres])
            s["auroc_presence"] = dict(jev=auroc(pj, pres), glance=auroc(pg, pres))
            s["ece_presence"] = dict(jev=ece(pj, pres), glance=ece(pg, pres))
        res[t] = s
    res["jev_failures"] = fails
    res["run"] = run_stats(run)
    return res


# ------------------------------------------------------------------------------------------------ GUI
def score_gui(root):
    gres = json.load(open(f"{REPO}/outputs/v1/eval_results.json"))["sets"]
    res = {}
    for b in ("ss2", "sspro", "venus"):
        if not os.path.exists(f"{root}/runs/{b}/requests.jsonl"):
            continue
        run = load_run(root, b)
        rows, fails = jev_items(run)

        def center_in(r):
            if r["pbox"] is None or r["box"] is None:
                return False
            cx, cy = (r["pbox"][0] + r["pbox"][2]) / 2, (r["pbox"][1] + r["pbox"][3]) / 2
            g = r["box"]
            return g[0] <= cx <= g[2] and g[1] <= cy <= g[3]
        pos = [r for r in rows if r["present"]]; neg = [r for r in rows if not r["present"]]
        g = {}
        gfn = f"{root}/data/glance/sprint_abs/rows_v1/{b}.jsonl.gz"
        if os.path.exists(gfn):
            with gzip.open(gfn, "rt") as f:
                for l in f:
                    r = json.loads(l); g[r["meta"]["id"]] = r
        gp = [g.get(r["meta"]["id"]) for r in pos]
        paired = np.array([x["center_in_box"] for x in gp]) if gp and all(x is not None for x in gp) else None
        s = dict(n=len(rows), n_pos=len(pos), n_refusal=len(neg),
                 click_acc=mean_ci([center_in(r) for r in pos], paired),
                 glance_click_acc_eval_results=gres[b]["raw"]["click_acc"],
                 gated_click_acc=float(np.mean([center_in(r) and r["p"] >= 0.5 for r in pos])),
                 jev_false_refusal_pos=float(np.mean([r["p"] < 0.5 for r in pos])),
                 jev_failures=fails, run=run_stats(run))
        if neg:
            gn = [g.get(r["meta"]["id"]) for r in neg]
            gpaired = None
            if all(x is not None for x in gn):
                gpaired = np.array([1 / (1 + math.exp(-x["p_logit"])) < 0.5 for x in gn])
            s["refusal_acc"] = mean_ci([r["p"] < 0.5 for r in neg], gpaired)
            s["glance_refusal_acc_eval_results"] = gres[b]["raw"]["refusal_acc"]
            s["auroc_presence"] = dict(jev=auroc([r["p"] for r in rows], [int(r["present"]) for r in rows]),
                                       glance_eval_results=gres[b]["raw"].get("auroc_presence"))
            n = len(rows)
            s["overall_micro"] = float((s["click_acc"]["v"] * len(pos) + s["refusal_acc"]["v"] * len(neg)) / n)
            s["glance_overall_micro"] = float((gres[b]["raw"]["click_acc"] * len(pos) + gres[b]["raw"]["refusal_acc"] * len(neg)) / n)
        groups = collections.defaultdict(list)
        for r in pos:
            m = r.get("meta") or {}
            groups[m.get("platform", "?") + "/" + str(m.get("ui_type") or m.get("task") or m.get("group") or "")].append(center_in(r))
        s["by_group"] = {k: dict(n=len(v), click_acc=float(np.mean(v)),
                                 glance=gres[b]["raw"]["by_group"].get(k, {}).get("click_acc")) for k, v in sorted(groups.items())}
        s["image_prep"] = dict(downscaled=sum(1 for _, raw in run if raw and any(m["scale"] < 1 for m in raw["img"].values())),
                               sent_max_side=max((max(m["sent_wh"]) for _, raw in run if raw for m in raw["img"].values()), default=None))
        res[b] = s
    return res


# ------------------------------------------------------------------------------------------------ probes
def wilson(k, n):
    if n == 0:
        return [float("nan")] * 3
    z = 1.96; p = k / n
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * p, 100 * max(0, c - h), 100 * min(1, c + h)]


def score_attr(root):
    run = load_run(root, "attr")
    rows, fails = jev_items(run)
    d = json.load(open(f"{REPO}/outputs/probes/ambiguity_attr/B_attribute_glance_v1.json"))
    gq = {(it["idx"], j): q for it in d["items"] for j, q in enumerate(it["questions"])}
    for r in rows:
        r["g"] = gq[(r["idx"], r["qi"])]
    res = {}
    for t in ("color", "action", "position", "size", "count"):
        neg = [r for r in rows if r["type"] == t and not r["present"]]
        pos = [r for r in rows if r["type"] == t and r["present"]] or [r for r in rows if r["type"] == "bare"]
        out = {}
        for name, getp in (("jev", lambda r: r["p"]), ("glance", lambda r: r["g"]["p"])):
            pn = [getp(r) for r in neg]; pp = [getp(r) for r in pos]
            out[name] = dict(FA_0_5=wilson(sum(x >= 0.5 for x in pn), len(pn)),
                             FR_0_5_pos=wilson(sum(x < 0.5 for x in pp), len(pp)),
                             auroc=auroc(pp + pn, [1] * len(pp) + [0] * len(pn)),
                             neg_boxed_on_object=wilson(sum((x >= 0.5) and iou_or0(r["pbox"] if name == "jev" else r["g"]["box"], r["box"]) >= 0.5
                                                            for x, r in zip(pn, neg)), len(pn)))
        out["n_pos"], out["n_neg"] = len(pos), len(neg)
        res[t] = out
    bare = [r for r in rows if r["type"] == "bare"]
    res["bare"] = dict(n=len(bare), jev_FR=wilson(sum(r["p"] < 0.5 for r in bare), len(bare)),
                       glance_FR=wilson(sum(r["g"]["p"] < 0.5 for r in bare), len(bare)),
                       jev_acc50=float(np.mean([iou_or0(r["pbox"], r["box"]) >= 0.5 for r in bare])),
                       glance_acc50=float(np.mean([r["g"]["iou_gt"] >= 0.5 for r in bare])))
    res["jev_failures"] = fails; res["run"] = run_stats(run)
    return res


def score_ambig(root):
    run = load_run(root, "ambig")
    d = json.load(open(f"{REPO}/outputs/probes/ambiguity_attr/A_ambiguity_glance_v1.json"))
    gq = {(it["image_id"], j): q for it in d["items"] for j, q in enumerate(it["questions"])}
    recs = []
    for req, raw in run:
        ans = raw["resp"]["answers"] if raw else {}
        for it in req["items"]:
            k = it["k"]; g = gq[(it["image_id"], it["qi"])]; W, H = it["W"], it["H"]
            r = dict(it, g=g)
            if it["tag"] == "count":
                a = ans.get(k, {}); r["probs"] = a.get("probabilities"); r["answer"] = a.get("choice")
            elif it["tag"] == "more_than_one":
                r["p_yes"] = (ans.get(k) or {}).get("noul")
            else:
                ab, ay = ans.get(k + "_box") or {}, ans.get(k + "_yes") or {}
                r["p"] = ay.get("noul", 0.0) if ay.get("type") == "noul" else 0.0
                b = ab.get("box")
                r["pbox"] = [b["left"] * W, b["top"] * H, b["right"] * W, b["bottom"] * H] if b else None
                r["conf"] = ab.get("confidence")
            recs.append(r)
    res = {}
    for kind in ("single", "multi"):
        for tag in ("bare_the", "bare_a", "plural", "leftmost", "rightmost", "biggest"):
            rr = [r for r in recs if r["kind"] == kind and r["tag"] == tag]
            if not rr:
                continue
            def hit_any(box):
                return box is not None and max(iou(box, b) for b in r_["gt_boxes"]) >= 0.5
            hj, hg = [], []
            for r_ in rr:
                hj.append(r_["pbox"] is not None and max(iou(r_["pbox"], b) for b in r_["gt_boxes"]) >= 0.5)
                hg.append(max(r_["g"]["iou_inst"]) >= 0.5)
            res[f"{kind}/{tag}"] = dict(n=len(rr),
                                        jev=dict(p_med=float(np.median([r["p"] for r in rr])),
                                                 rejected=float(np.mean([r["p"] < 0.5 for r in rr])),
                                                 hit_any_instance=float(np.mean(hj))),
                                        glance=dict(p_med=float(np.median([r["g"]["p"] for r in rr])),
                                                    rejected=float(np.mean([r["g"]["p"] < 0.5 for r in rr])),
                                                    hit_any_instance=float(np.mean(hg))))
    # "more than one?" as a single-vs-multi detector; count accuracy
    m1 = [r for r in recs if r["tag"] == "more_than_one"]
    y = [int(r["kind"] == "multi") for r in m1]
    res["more_than_one_auroc"] = dict(n=len(m1), jev=auroc([r["p_yes"] if r["p_yes"] is not None else 0.5 for r in m1], y),
                                      glance=auroc([r["g"]["p_yes"] for r in m1], y))
    cnt = [r for r in recs if r["tag"] == "count"]

    def cnt_ok(ans, n):
        return ans == (str(n) if n < 6 else "6 or more")
    res["count_acc"] = dict(n=len(cnt), jev=float(np.mean([cnt_ok(r["answer"], r["n_inst"]) for r in cnt])),
                            glance=float(np.mean([cnt_ok(r["g"]["answer"], r["n_inst"]) for r in cnt])))
    res["run"] = run_stats(run)
    return res


def score_solo(root):
    """Multi-question control: same RefCOCO-val items, one expression per request vs all of an image's expressions
    in one request (the refcoco_val run)."""
    solo, _ = jev_items(load_run(root, "solo"))
    grp = {}
    for req, raw in load_run(root, "refcoco_val"):
        rows, _ = jev_items([(req, raw)])
        for r in rows:
            grp[(req["id"], r["k"])] = r
    s_hit, g_hit, s_p, g_p = [], [], [], []
    for r in solo:
        g = grp[(r["grouped_id"], r["grouped_k"])]
        s_hit.append(iou_or0(r["pbox"], r["box"]) >= 0.5); g_hit.append(iou_or0(g["pbox"], g["box"]) >= 0.5)
        s_p.append(r["p"]); g_p.append(g["p"])
    s_hit, g_hit, s_p, g_p = map(np.array, (s_hit, g_hit, s_p, g_p))
    d = mean_ci(s_hit, g_hit)
    return dict(n=len(solo), acc50_solo_vs_grouped=dict(solo=d["v"], grouped=d["glance"], diff=d["diff"], diff_ci=d["diff_ci"]),
                false_refusal_solo=float((s_p < 0.5).mean()), false_refusal_grouped=float((g_p < 0.5).mean()),
                p_mean_abs_diff=float(np.abs(s_p - g_p).mean()),
                note="'glance' fields inside mean_ci here mean the grouped Jev run, not Glance")


# ------------------------------------------------------------------------------------------------ OpenDecisions
def score_od(root):
    import subprocess
    run = load_run(root, "od")
    od = {json.loads(l)["id"]: json.loads(l) for l in open(f"{root}/data/od/opendecisions.jsonl")}
    fn = f"{root}/runs/od/predictions.jsonl"
    fails = collections.Counter(); n = 0
    with open(fn, "w") as f:
        for req, raw in run:
            it = req["items"][0]; t = it["type"]; labs = [str(x) for x in it["labels"]]
            a = ((raw or {}).get("resp", {}).get("answers", {}) or {}).get("answer")
            if not a or a.get("type") != t:
                fails[(a or {}).get("type", "missing") if raw else "request_failed"] += 1
                continue
            if t == "noul":
                p = float(a["noul"]); probs = {"yes": p, "no": 1 - p}
            else:
                pr = a.get("probabilities") or {}
                if t == "score" and not set(pr) <= set(labs):
                    crit = [str(c) for c in od[it["od_id"]]["question"]["criteria"]]
                    pr = {str(crit.index(k)) if k in crit else k: v for k, v in pr.items()}
                probs = {l: float(pr.get(l, 0.0)) for l in labs}
                s = sum(probs.values())
                if s <= 0:
                    fails["zero_mass"] += 1; continue
                probs = {k: v / s for k, v in probs.items()}  # API rounds to 2 decimals
            f.write(json.dumps(dict(id=it["od_id"], probs=probs)) + "\n"); n += 1
    out = subprocess.run([sys.executable, "-I", f"{root}/data/od/score.py", fn, "--data", f"{root}/data/od/opendecisions.jsonl",
                          "--json"], capture_output=True, text=True)
    try:
        sc = json.loads(out.stdout)
    except ValueError:
        sc = dict(error=(out.stdout + out.stderr)[-2000:])
    glance = json.load(open(f"{REPO}/outputs/jev/od_clipped.json")) if os.path.exists(f"{REPO}/outputs/jev/od_clipped.json") else None
    return dict(jev=sc, glance_v1_clipped=glance, predicted=n, jev_failures=dict(fails), run=run_stats(run))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--root", default=os.path.expanduser("~/jev"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    names = ["grefcoco", "refcoco8", "robust", "gui", "attr", "ambig", "od", "solo"] if a.bench == "all" else [a.bench]
    res = {}
    for nm in names:
        res[nm] = globals()[f"score_{nm}"](a.root)
        print(nm, json.dumps(res[nm], default=str)[:3000], flush=True)
    if a.out:
        json.dump(res if len(res) > 1 else res[names[0]], open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
