"""Offline analysis for the absence/confidence sprint (CPU).

python scripts/sprint_abs/analyze.py --rows <dir with *.jsonl.gz from dump.py> --venus <dir with official venus jsons> \
    --tag v1 --out outputs/sprint_abs/v1

All calibrators / decision parameters are fitted ONLY on:
  natural: calib_natural (600 held-out training-distribution COCO images, 28% absent)  [+ grefcalib for GREC set decoding:
           gRefCOCO *train* expressions on the same held-out images]
  GUI:     guisyn "fit" apps (guisyn is outside the v1/v1.1 mixtures, no eval image); the other guisyn apps are a
           held-out GUI check.
Risk-controlled (RCPS / LTT) thresholds use a random half of the target benchmark as the labelled calibration split
(exchangeability needed for the guarantee) and are tested on the other half, repeated over random image-level splits;
the "transfer" variant calibrates on the training-distribution splits instead (no guarantee under shift).
"""
import argparse, collections, json, math, os, sys, zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (load_rows, sig, logit, auroc, ece, calib_metrics, aurc, make_cal, MultiPlatt, Logistic, prior_shift,
                 em_prior, grec_metrics, giou_mat, rcps_threshold, sgr_threshold, conformal_absent_threshold, cluster_boot, cp_upper)

NAT_EVAL = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocoplus_testA", "refcocoplus_testB",
            "refcocog_val", "refcocog_test", "grec_val", "grec_testA", "grec_testB"]
GUI_EVAL = ["ss2", "venus"]
CAL_METHODS = ["temp", "platt", "beta", "iso"]


# ----------------------------------------------------------------------------------------------- row helpers
def arr(rows, f):
    return np.array([f(r) for r in rows], float)


def best_iou(r):
    if "iou" in r:
        return r["iou"]
    gb = r.get("gt_boxes") or []
    if not gb:
        return 0.0
    iou, _ = giou_mat([r["box"]], gb)
    return float(iou.max())


def click_ok(r):
    if r.get("center_in_box") is not None:
        return bool(r["center_in_box"])
    b = r["box"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in (r.get("gt_boxes") or []))


def nwords(r):
    return len(r["text"].split())


def group_id(r):
    return r.get("img") or json.dumps(r["wh"])


def guisyn_split(r):
    app = (r.get("meta") or {}).get("app", "x")
    return "fit" if zlib.crc32(app.encode()) % 2 == 0 else "test"


def feats_kappa(rows):
    """observable features for the multi-feature kappa recalibrator"""
    return np.stack([arr(rows, lambda r: r["iou_logit"]), logit(arr(rows, lambda r: r["center_top1"])),
                     arr(rows, lambda r: r["center_ent"]), np.log(arr(rows, lambda r: np.mean(r["edge_sd"])) + 1e-4),
                     np.log(arr(rows, lambda r: r["n_vis"])), np.log1p(arr(rows, nwords))], 1)


def feats_presence(rows):
    """model read-outs of the same forward only (no query/image metadata that could encode how a dataset built its
    negatives): presence logit, center-distribution peak and entropy, IoU-head logit, edge spread."""
    return np.stack([arr(rows, lambda r: r["p_logit"]),
                     logit(arr(rows, lambda r: r["center_top1"])), arr(rows, lambda r: r["center_ent"]),
                     arr(rows, lambda r: r["iou_logit"]), np.log(arr(rows, lambda r: np.mean(r["edge_sd"])) + 1e-4)], 1)


# ----------------------------------------------------------------------------------------------- presence
def fit_presence(cal_rows):
    z, y = arr(cal_rows, lambda r: r["p_logit"]), arr(cal_rows, lambda r: r["present"])
    cals = {m: make_cal(m).fit(z, y) for m in CAL_METHODS}
    cals["multi"] = MultiPlatt().fit(feats_presence(cal_rows), y)
    return cals, float(y.mean())


def presence_probs(cals, rows):
    z = arr(rows, lambda r: r["p_logit"])
    out = {"raw": sig(z)}
    for m in CAL_METHODS:
        out[m] = cals[m](z)
    out["multi"] = cals["multi"](feats_presence(rows))
    return out


def presence_table(probs, rows, pi_src, label="all"):
    y = arr(rows, lambda r: r["present"])
    res = {}
    for m, p in probs.items():
        d = calib_metrics(p, y)
        d["auroc"] = auroc(p, y)
        if (y == 0).any():
            d["N_acc"] = float((p[y == 0] < 0.5).mean())
        if (y == 1).any():
            d["recall_present"] = float((p[y == 1] >= 0.5).mean())
        if m != "raw":
            pi_hat, q = em_prior(p, pi_src)
            dq = calib_metrics(q, y); dq["pi_hat"] = pi_hat
            if (y == 0).any():
                dq["N_acc"] = float((q[y == 0] < 0.5).mean())
            if (y == 1).any():
                dq["recall_present"] = float((q[y == 1] >= 0.5).mean())
            d["em_prior"] = dq
        res[m] = d
    return res


# ----------------------------------------------------------------------------------------------- kappa
def fit_kappa(cal_rows, event):
    rr = [r for r in cal_rows if r["present"] and (r.get("gt_box") is not None)]
    z, y = arr(rr, lambda r: r["iou_logit"]), arr(rr, event)
    cals = {m: make_cal(m).fit(z, y) for m in CAL_METHODS}
    cals["multi"] = MultiPlatt().fit(feats_kappa(rr), y)
    return cals


def kappa_probs(cals, rows):
    z = arr(rows, lambda r: r["iou_logit"])
    out = {"raw": sig(z)}
    for m in CAL_METHODS:
        out[m] = cals[m](z)
    out["multi"] = cals["multi"](feats_kappa(rows))
    return out


def ev_iou(r):
    return float(best_iou(r) >= 0.5)


def ev_click(r):
    return float(click_ok(r))


# ----------------------------------------------------------------------------------------------- GREC set decoding
def nms_cands(r, nms_iou=0.5, max_keep=6):
    """candidate boxes [p_center, mass_in_box, iou_logit, x1, y1, x2, y2, remaining_mass]: the suppression peaks when
    dumped (peak k = most probable patch outside the boxes of peaks < k), else the top-K center patches."""
    if r.get("peaks"):
        c = np.array([[p[0], p[1], p[3]] + p[4:8] + [p[2]] for p in r["peaks"]], float)
    else:
        c = np.array([x + [x[1]] for x in r["cand"]], float)
    boxes = c[:, 3:7]
    keep = [0]
    for j in range(1, len(c)):
        if len(keep) >= max_keep:
            break
        iou, _ = giou_mat(boxes[j:j + 1], boxes[keep])
        # also drop candidates whose box center lies inside an already kept box (same object, other patch)
        if iou.max() < nms_iou:
            keep.append(j)
    return c[keep]


def extra_feats(r, kept):
    c0 = kept[0]
    out = []
    for k, c in enumerate(kept[1:], start=1):
        iou0, _ = giou_mat([c[3:7]], [c0[3:7]])
        out.append([math.log(c[1] + 1e-6), math.log(c[1] / max(c0[1], 1e-6) + 1e-6), math.log(c[0] / max(c0[0], 1e-6) + 1e-6),
                    c[2], float(k), r["p_logit"], math.log(c[0] + 1e-6), math.log(c[7] + 1e-6), float(iou0[0, 0])])
    return np.array(out, float).reshape(-1, 9)


def extra_labels(r, kept):
    gts = r.get("gt_boxes") or []
    if not gts:
        return np.zeros(len(kept) - 1)
    _, g = giou_mat(kept[:, 3:7], gts)
    used = set()
    # greedy match top-1 first (it is always output when present)
    lab = []
    for i in range(len(kept)):
        j = int(np.argmax(np.where([k in used for k in range(len(gts))], -9, g[i])))
        hit = g[i, j] >= 0.5 and j not in used
        if hit:
            used.add(j)
        if i > 0:
            lab.append(float(hit))
    return np.array(lab)


def fit_extra(rows):
    X, Y = [], []
    for r in rows:
        if not r["present"]:
            continue
        kept = nms_cands(r)
        if len(kept) > 1:
            X.append(extra_feats(r, kept)); Y.append(extra_labels(r, kept))
    X, Y = np.concatenate(X), np.concatenate(Y)
    return MultiPlatt().fit(X, Y), dict(n=int(len(Y)), pos=float(Y.mean()))


def grec_samples(rows, pres_p, mode="single", extra=None, t_extra=None):
    """pred set per row: top-1 box scored with presence p; mode multi adds NMS'd extra candidates scored p * h_j."""
    S = []
    for r, p in zip(rows, pres_p):
        gts = r.get("gt_boxes") or ([] if not r["present"] else [r["gt_box"]])
        if not r["present"]:
            gts = []
        boxes, scores = [r["box"]], [float(p)]
        if mode == "multi":
            kept = nms_cands(r)
            if len(kept) > 1:
                h = extra(extra_feats(r, kept))
                for c, hj in zip(kept[1:], h):
                    boxes.append(list(c[3:7])); scores.append(float(p * hj) if t_extra is None else (float(p) if hj >= t_extra else 0.0))
        S.append((boxes, scores, gts))
    return S


# ----------------------------------------------------------------------------------------------- risk control
def rc_experiment(score, loss, groups, alphas, delta=0.1, mode="selective", reps=200, seed=0, transfer=None, grid=None):
    """Repeated 50/50 image-level split: calibrate threshold on one half (RCPS/LTT), report coverage and realized risk
    on the other half. transfer=(score_cal, loss_cal): calibrate on that external set instead (single run, full test).
    grid: fixed threshold sequence (from the independent training-distribution split)."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups); uniq = np.unique(g)
    out = {}
    for a in alphas:
        covs, risks, viol = [], [], []
        if transfer is not None:
            t = sgr_threshold(transfer[0], transfer[1], a, delta) if mode == "selective" else rcps_threshold(transfer[0], transfer[1], a, delta, mode, grid=grid)
            acc = score >= t
            cov = float(acc.mean())
            risk = float(loss[acc].mean()) if (mode == "selective" and acc.any()) else (float((loss * acc).mean()) if mode == "joint" else 0.0)
            out[str(a)] = dict(threshold=float(t), coverage=cov, risk=risk, violated=bool(risk > a))
            continue
        for _ in range(reps):
            cal_g = set(rng.choice(uniq, len(uniq) // 2, replace=False))
            m = np.array([x in cal_g for x in g])
            t = sgr_threshold(score[m], loss[m], a, delta) if mode == "selective" else rcps_threshold(score[m], loss[m], a, delta, mode, grid=grid)
            acc = (score[~m] >= t)
            covs.append(acc.mean())
            if mode == "selective":
                risks.append(loss[~m][acc].mean() if acc.any() else 0.0)
            else:
                risks.append((loss[~m] * acc).mean())
            viol.append(risks[-1] > a)
        out[str(a)] = dict(coverage=float(np.mean(covs)), coverage_sd=float(np.std(covs)), risk=float(np.mean(risks)),
                           risk_p90=float(np.percentile(risks, 90)), violation_rate=float(np.mean(viol)))
    return out


def conformal_absent_experiment(score, absent, groups, alphas, reps=200, seed=0, transfer=None):
    """Guarantee P(accept | absent) <= alpha by split conformal on absent calibration items; report present acceptance."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups); uniq = np.unique(g)
    out = {}
    for a in alphas:
        if transfer is not None:
            q = conformal_absent_threshold(transfer, a)
            acc = score > q
            out[str(a)] = dict(threshold=float(q), false_accept_absent=float(acc[absent].mean()),
                               accept_present=float(acc[~absent].mean()))
            continue
        fa, ap = [], []
        for _ in range(reps):
            cal_g = set(rng.choice(uniq, len(uniq) // 2, replace=False))
            m = np.array([x in cal_g for x in g])
            q = conformal_absent_threshold(score[m & absent], a)
            acc = score[~m] > q
            fa.append(acc[absent[~m]].mean()); ap.append(acc[~absent[~m]].mean())
        out[str(a)] = dict(false_accept_absent=float(np.mean(fa)), false_accept_sd=float(np.std(fa)),
                           accept_present=float(np.mean(ap)))
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--venus", default="")
    ap.add_argument("--tag", default="v1"); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {}
    for fn in sorted(os.listdir(a.rows)):
        if fn.endswith(".jsonl.gz"):
            R[fn[:-9]] = load_rows(os.path.join(a.rows, fn))
    print({k: len(v) for k, v in R.items()}, flush=True)
    res = dict(tag=a.tag, n_rows={k: len(v) for k, v in R.items()})

    # ---------------- presence calibration
    cal_nat = R["calib_natural"]
    pc_nat, pi_nat = fit_presence(cal_nat)
    gs = R.get("guisyn", [])
    gs_fit = [r for r in gs if guisyn_split(r) == "fit"]; gs_test = [r for r in gs if guisyn_split(r) == "test"]
    pc_gui, pi_gui = fit_presence(gs_fit) if gs_fit else (pc_nat, pi_nat)
    res["calibrators"] = dict(natural=dict(T=pc_nat["temp"].T, platt=list(pc_nat["platt"].m.b), pi_src=pi_nat, n=len(cal_nat)),
                              gui=dict(T=pc_gui["temp"].T, platt=list(pc_gui["platt"].m.b), pi_src=pi_gui, n_fit=len(gs_fit),
                                       n_test=len(gs_test)))
    pres = {}
    for name in NAT_EVAL + ["calib_natural", "grefcalib"]:
        if name not in R:
            continue
        rows = R[name]
        P = presence_probs(pc_nat, rows)
        if not any(not r["present"] for r in rows):
            continue  # all-present sets: presence calibration is degenerate (reported via box confidence below)
        pres[name] = dict(all=presence_table(P, rows, pi_nat))
        if name.startswith("grec_"):
            m = np.array([r["n_targets"] != 1 for r in rows])  # gRefCOCO no-target + multi-target only
            if m.any() and (~m).any():
                sub = [r for r, k in zip(rows, m) if k]
                pres[name]["gref_only"] = presence_table({k: v[m] for k, v in P.items()}, sub, pi_nat)
    for name, rows in [("guisyn_test", gs_test), ("venus", R.get("venus", []))]:
        if not rows:
            continue
        pres[name] = dict(natural_fit=presence_table(presence_probs(pc_nat, rows), rows, pi_nat),
                          gui_fit=presence_table(presence_probs(pc_gui, rows), rows, pi_gui))
    res["presence"] = pres

    # ---------------- kappa calibration (natural: IoU>=.5 ; GUI: click-in-box)
    kc_nat = fit_kappa(cal_nat, ev_iou)
    kc_gui = fit_kappa(gs_fit, ev_click) if gs_fit else kc_nat
    kap = {}
    for name in NAT_EVAL + ["calib_natural"]:
        if name not in R:
            continue
        rr = [r for r in R[name] if r["present"] and r.get("gt_box") is not None]
        K = kappa_probs(kc_nat, rr); y = arr(rr, ev_iou)
        kap[name] = {m: dict(**calib_metrics(p, y), auroc=auroc(p, y), aurc=aurc(p, 1 - y)) for m, p in K.items()}
    for name, rows in [("guisyn_test", gs_test), ("ss2", R.get("ss2", [])), ("venus", R.get("venus", []))]:
        rr = [r for r in rows if r["present"] and r.get("gt_box") is not None]
        if not rr:
            continue
        y = arr(rr, ev_click)
        kap[name] = {}
        for src, kc in (("natural_fit", kc_nat), ("gui_fit", kc_gui)):
            K = kappa_probs(kc, rr)
            kap[name][src] = {m: dict(**calib_metrics(p, y), auroc=auroc(p, y), aurc=aurc(p, 1 - y)) for m, p in K.items()}
    res["kappa"] = kap

    # ---------------- box confidence g*kappa vs (present and IoU>=.5): the v0->v1 regression metric
    box = {}
    for name in NAT_EVAL:
        if name not in R:
            continue
        rows = [r for r in R[name] if (r.get("gt_box") is not None) or not r["present"]]
        y = arr(rows, lambda r: float(r["present"] and best_iou(r) >= 0.5))
        P = presence_probs(pc_nat, rows); K = kappa_probs(kc_nat, rows)
        d = {}
        d["raw"] = calib_metrics(P["raw"] * K["raw"], y)
        d["evalpy_T"] = calib_metrics(P["temp"] * K["raw"], y)
        for pm in ("platt", "iso"):
            for km in ("platt", "iso", "multi"):
                d[f"g:{pm}|k:{km}"] = calib_metrics(P[pm] * K[km], y)
                pi_hat, q = em_prior(P[pm], pi_nat)
                d[f"g:{pm}+em|k:{km}"] = dict(calib_metrics(q * K[km], y), pi_hat=pi_hat)
        d["kappa_only_raw"] = calib_metrics(K["raw"], y)
        box[name] = d
    res["box_conf"] = box

    # ---------------- GREC official protocol
    ex_model, ex_info = fit_extra(R["grefcalib"]) if "grefcalib" in R else (None, None)
    res["grec_extra_model"] = ex_info
    # thresholds tuned on grefcalib (held-out train-split expressions on held-out images) to maximise Pr@F1=1
    tune = {}
    if "grefcalib" in R:
        rows = R["grefcalib"]; pp = presence_probs(pc_nat, rows)["platt"]
        best = None
        S0 = grec_samples(rows, pp, "multi", ex_model)
        for tg in np.arange(0.3, 0.91, 0.05):
            for tx in np.arange(0.2, 0.96, 0.05):
                S = [(b, [s if k == 0 else (pp_ if s >= tx * pp_ else 0.0) for k, s in enumerate(sc)], g)
                     for (b, sc, g), pp_ in zip(S0, pp)]
                m = grec_metrics(S, thresh_score=tg)
                if best is None or m["pr_f1"] > best[0]:
                    best = (m["pr_f1"], float(tg), float(tx))
        tune = dict(pr_f1_grefcalib=best[0], t_presence=best[1], t_extra=best[2])
        # single-box presence threshold
        bs = max(((grec_metrics(grec_samples(rows, pp), thresh_score=t)["pr_f1"], float(t)) for t in np.arange(0.3, 0.91, 0.05)))
        tune["single_t_presence"], tune["single_pr_f1_grefcalib"] = bs[1], bs[0]
    res["grec_tuned_on_grefcalib"] = tune
    grec = {}
    for name in ["grec_val", "grec_testA", "grec_testB", "grefcalib"]:
        if name not in R:
            continue
        rows = R[name]
        P = presence_probs(pc_nat, rows)
        groups = [group_id(r) for r in rows]
        d = {}
        variants = {
            "single_raw@0.7(official)": (grec_samples(rows, P["raw"]), 0.7),
            "single_raw@0.5": (grec_samples(rows, P["raw"]), 0.5),
            "single_platt@0.7": (grec_samples(rows, P["platt"]), 0.7),
        }
        if ex_model is not None:
            variants["multi_platt@0.7(official)"] = (grec_samples(rows, P["platt"], "multi", ex_model), 0.7)
            variants["multi_raw@0.7(official)"] = (grec_samples(rows, P["raw"], "multi", ex_model), 0.7)
            if tune:
                S = grec_samples(rows, P["platt"], "multi", ex_model)
                S = [(b, [s if k == 0 else (p_ if s >= tune["t_extra"] * p_ else 0.0) for k, s in enumerate(sc)], g)
                     for (b, sc, g), p_ in zip(S, P["platt"])]
                variants["multi_platt@tuned"] = (S, tune["t_presence"])
                variants["single_platt@tuned"] = (grec_samples(rows, P["platt"]), tune["single_t_presence"])
        for k, (S, t) in variants.items():
            m = grec_metrics(S, thresh_score=t)
            m_iou = grec_metrics(S, thresh_score=t, use_giou=False)
            m["pr_f1_plainIoU"] = m_iou["pr_f1"]
            ok = None
            if a.boot and name != "grefcalib":
                from lib import grec_sample
                okv = np.array([grec_sample(*s, thresh_score=t)[0] for s in S], float)
                m["pr_f1_ci"] = cluster_boot(groups, lambda idx: okv[idx].mean(), B=a.boot)
            d[k] = m
        if a.boot and ex_model is not None and name != "grefcalib":
            from lib import grec_sample
            o1 = np.array([grec_sample(*s, thresh_score=0.7)[0] for s in variants["single_raw@0.7(official)"][0]], float)
            o2 = np.array([grec_sample(*s, thresh_score=0.7)[0] for s in variants["multi_platt@0.7(official)"][0]], float)
            d["paired_multi_minus_single_official"] = cluster_boot(groups, lambda idx: o2[idx].mean() - o1[idx].mean(), B=a.boot)
        grec[name] = d
    res["grec"] = grec

    # ---------------- risk control (deployment): accept/abstain with guaranteed false-accept / misclick
    alphas = [0.01, 0.02, 0.05, 0.1]
    rc = {}
    Pn_cal = presence_probs(pc_nat, cal_nat); Kn_cal = kappa_probs(kc_nat, cal_nat)
    cal_loss_nat = 1 - arr(cal_nat, lambda r: float(r["present"] and best_iou(r) >= 0.5))
    if gs_fit:
        Pg_cal = presence_probs(pc_gui, gs_fit); Kg_cal = kappa_probs(kc_gui, gs_fit)
        cal_loss_gui = 1 - arr(gs_fit, lambda r: float(r["present"] and click_ok(r)))
    for name, dom in [("refcoco_val", "nat"), ("refcocog_test", "nat"), ("grec_val", "nat"), ("grec_testA", "nat"),
                      ("grec_testB", "nat"), ("ss2", "gui"), ("venus", "gui")]:
        if name not in R:
            continue
        rows = R[name]
        if dom == "nat":
            P, K = presence_probs(pc_nat, rows), kappa_probs(kc_nat, rows)
            correct = arr(rows, lambda r: float(r["present"] and best_iou(r) >= 0.5))
            tr = dict(g=Pn_cal["platt"], gk=Pn_cal["platt"] * Kn_cal["multi"], loss=cal_loss_nat,
                      absent=Pn_cal["platt"][arr(cal_nat, lambda r: r["present"]) == 0])
        else:
            P, K = presence_probs(pc_gui, rows), kappa_probs(kc_gui, rows)
            correct = arr(rows, lambda r: float(r["present"] and click_ok(r)))
            tr = dict(g=Pg_cal["platt"], gk=Pg_cal["platt"] * Kg_cal["multi"], loss=cal_loss_gui,
                      absent=Pg_cal["platt"][arr(gs_fit, lambda r: r["present"]) == 0]) if gs_fit else None
        loss = 1 - correct
        groups = [group_id(r) for r in rows]
        absent = arr(rows, lambda r: r["present"]) == 0
        scores = {"g": P["platt"], "g*k": P["platt"] * K["multi"]}
        d = dict(n=len(rows), absent_rate=float(absent.mean()), base_correct=float(correct.mean()),
                 default_rule=dict(coverage=float((P["raw"] >= 0.5).mean()),
                                   selective_risk=float(loss[P["raw"] >= 0.5].mean()),
                                   false_accept_absent=float((P["raw"][absent] >= 0.5).mean()) if absent.any() else None))
        for sk, s in scores.items():
            ts = None if tr is None else (tr["g"] if sk == "g" else tr["gk"])
            grid = None if ts is None else np.unique(np.quantile(ts, 1 - np.linspace(0.01, 1.0, 300)))
            d[f"selective[{sk}]"] = rc_experiment(s, loss, groups, alphas, reps=a.reps, grid=grid)
            d[f"joint[{sk}]"] = rc_experiment(s, loss, groups, alphas, reps=a.reps, mode="joint", grid=grid)
            if tr is not None:
                d[f"selective_transfer[{sk}]"] = rc_experiment(s, loss, groups, alphas, transfer=(ts, tr["loss"]), grid=grid)
            d[f"aurc[{sk}]"] = aurc(s, loss)
        if absent.any():
            d["absent_conformal[g]"] = conformal_absent_experiment(P["platt"], absent, groups, alphas, reps=a.reps)
            if tr is not None:
                d["absent_conformal_transfer[g]"] = conformal_absent_experiment(P["platt"], absent, groups, alphas, transfer=tr["absent"])
        if a.boot:
            gl = scores["g"]; gkl = scores["g*k"]
            d["aurc_paired_gk_minus_g"] = cluster_boot(groups, lambda idx: aurc(gkl[idx], loss[idx]) - aurc(gl[idx], loss[idx]), B=min(a.boot, 300))
        rc[name] = d
    res["risk_control"] = rc

    # ---------------- Prop. 1 two-knob rule (plug-in, calibrated probabilities; no tuning) vs one-knob thresholds
    p1 = {}
    costs = [dict(c_H=1, c_W=1, c_R=0.5), dict(c_H=1, c_W=2, c_R=0.5), dict(c_H=1, c_W=1, c_R=1.0), dict(c_H=2, c_W=1, c_R=0.5),
             dict(c_H=1, c_W=4, c_R=0.5)]
    for name, dom in [("refcoco_val", "nat"), ("refcocog_test", "nat"), ("grec_val", "nat"), ("grec_testA", "nat"), ("grec_testB", "nat"),
                      ("ss2", "gui"), ("venus", "gui")]:
        if name not in R:
            continue
        rows = R[name]
        P, K = (presence_probs(pc_nat, rows), kappa_probs(kc_nat, rows)) if dom == "nat" else (presence_probs(pc_gui, rows), kappa_probs(kc_gui, rows))
        ok = arr(rows, (lambda r: float(r["present"] and best_iou(r) >= 0.5)) if dom == "nat" else (lambda r: float(r["present"] and click_ok(r))))
        pr = arr(rows, lambda r: float(r["present"]))
        eta, kap_ = P["platt"], K["multi"]
        groups = [group_id(r) for r in rows]
        dd = {}
        for c in costs:
            def cost(acc, idx=None):
                idx = np.arange(len(rows)) if idx is None else idx
                A = acc[idx]
                return float(np.mean(np.where(A, np.where(pr[idx] == 0, c["c_H"], np.where(ok[idx] == 1, 0.0, c["c_W"])),
                                              np.where(pr[idx] == 1, c["c_R"], 0.0))))
            two = (c["c_R"] - c["c_W"] + c["c_H"]) * eta + c["c_W"] * eta * kap_ > c["c_H"]
            one_default = P["raw"] >= 0.5
            # one-knob threshold on eta chosen on the training-distribution calibration split (same costs)
            if dom == "nat":
                ce, ck, cok, cpr = Pn_cal["platt"], Kn_cal["multi"], 1 - cal_loss_nat, arr(cal_nat, lambda r: float(r["present"]))
            else:
                ce, ck, cok, cpr = Pg_cal["platt"], Kg_cal["multi"], 1 - cal_loss_gui, arr(gs_fit, lambda r: float(r["present"]))
            def ccost(acc):
                return float(np.mean(np.where(acc, np.where(cpr == 0, c["c_H"], np.where(cok == 1, 0.0, c["c_W"])), np.where(cpr == 1, c["c_R"], 0.0))))
            ts = np.linspace(0.05, 0.95, 91)
            t_star = ts[int(np.argmin([ccost(ce >= t) for t in ts]))]
            one_tuned = eta >= t_star
            s_star = ts[int(np.argmin([ccost(ce * ck >= t) for t in ts]))]
            prod_tuned = eta * kap_ >= s_star
            t_or = ts[int(np.argmin([cost(eta >= t) for t in ts]))]   # oracle one-knob (tuned on the test set itself)
            e = dict(two_knob_plugin=cost(two), one_knob_default=cost(one_default), one_knob_tuned=cost(one_tuned),
                     product_tuned=cost(prod_tuned), one_knob_oracle=cost(eta >= t_or), t_star=float(t_star), t_oracle=float(t_or),
                     coverage_two=float(two.mean()), coverage_one_tuned=float(one_tuned.mean()))
            if a.boot:
                e["paired_two_minus_one_tuned"] = cluster_boot(groups, lambda idx: cost(two, idx) - cost(one_tuned, idx), B=min(a.boot, 500))
            dd[json.dumps(c)] = e
        p1[name] = dd
    res["prop1"] = p1

    # ---------------- bootstrap CIs for headline presence numbers
    if a.boot:
        hb = {}
        for name in ["grec_val", "grec_testA", "grec_testB", "venus"]:
            if name not in R:
                continue
            rows = R[name]
            pcs = pc_gui if name == "venus" else pc_nat
            pis = pi_gui if name == "venus" else pi_nat
            P = presence_probs(pcs, rows)
            y = arr(rows, lambda r: r["present"]); groups = [group_id(r) for r in rows]
            sel = np.ones(len(rows), bool) if name != "grec_val" else np.array([r["n_targets"] != 1 for r in rows])
            pr, pp_, yy, gg = P["raw"][sel], P["platt"][sel], y[sel], [g for g, s in zip(groups, sel) if s]
            hb[name] = dict(
                auroc=cluster_boot(groups, lambda idx: auroc(P["raw"][idx], y[idx]), B=a.boot),
                N_acc=cluster_boot(groups, lambda idx: float((P["raw"][idx][y[idx] == 0] < 0.5).mean()), B=a.boot),
                ece_raw=cluster_boot(groups, lambda idx: ece(P["raw"][idx], y[idx]), B=a.boot),
                ece_platt=cluster_boot(groups, lambda idx: ece(P["platt"][idx], y[idx]), B=a.boot),
                ece_paired_platt_minus_raw=cluster_boot(groups, lambda idx: ece(P["platt"][idx], y[idx]) - ece(P["raw"][idx], y[idx]), B=a.boot),
                ece_sub_raw=cluster_boot(gg, lambda idx: ece(pr[idx], yy[idx]), B=a.boot),
                ece_sub_platt=cluster_boot(gg, lambda idx: ece(pp_[idx], yy[idx]), B=a.boot),
            )
        res["boot_presence"] = hb
        bb = {}
        for name in ["refcoco_val", "refcocog_test"]:
            if name not in R:
                continue
            rows = [r for r in R[name] if r.get("gt_box") is not None]
            y = arr(rows, lambda r: float(best_iou(r) >= 0.5)); groups = [group_id(r) for r in rows]
            P, K = presence_probs(pc_nat, rows), kappa_probs(kc_nat, rows)
            _, q = em_prior(P["platt"], pi_nat)
            c_raw, c_fix = P["raw"] * K["raw"], q * K["platt"]
            bb[name] = dict(raw=cluster_boot(groups, lambda idx: ece(c_raw[idx], y[idx]), B=a.boot),
                            fixed=cluster_boot(groups, lambda idx: ece(c_fix[idx], y[idx]), B=a.boot),
                            paired=cluster_boot(groups, lambda idx: ece(c_fix[idx], y[idx]) - ece(c_raw[idx], y[idx]), B=a.boot),
                            kappa_auroc=cluster_boot(groups, lambda idx: auroc(K["raw"][idx], y[idx]), B=a.boot))
        res["boot_box_conf"] = bb

    # ---------------- official VenusBench-GD scorer outputs
    if a.venus and os.path.isdir(a.venus):
        vo = {}
        for cfg in sorted(os.listdir(a.venus)):
            cd = os.path.join(a.venus, cfg)
            if not os.path.isdir(cd):
                continue
            t = {}
            for fn in sorted(os.listdir(cd)):
                if fn.endswith(".json"):
                    o = json.load(open(os.path.join(cd, fn)))["metrics"]["overall"]
                    t[fn[:-5]] = dict(acc=o["action_acc"], n=o["num_total"], correct=o["num_correct_action"],
                                      wrong_format=o["wrong_format_num"])
            if len(t) < 6:
                continue
            basic = [k for k in t if k.split("_")[0] in ("element", "spatial", "visual")]
            adv = [k for k in t if k not in basic]
            sm = lambda ks: sum(t[k]["correct"] for k in ks) / sum(t[k]["n"] for k in ks)
            t["overall"] = dict(acc=sm(list(t)), n=sum(t[k]["n"] for k in t), basic=sm(basic), advanced=sm(adv))
            vo[cfg] = t
        res["venus_official"] = vo
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(a.out, "results.json"))


if __name__ == "__main__":
    main()
