"""Score Glance-X rows (scripts/glancex/dump_rows.py) offline, CPU only, under both protocols.

  python scripts/glancex/score.py ROWS_DIR OUT.json [--boot 1000]

Sections of OUT.json:
  refcoco     Acc@0.5 of the single box on the 8 RefCOCO/+/g splits (= glancex.evaluate acc50 = our eval.py acc50)
  gref_ours   gRefCOCO val_merged / testA / testB with our eval.py definitions: N_acc (= Glance-X reject_rate,
              P<0.5 on no-target), T_acc_single (gated), acc50 single (ungated), present_recall_multi, AUROC,
              ECE (15 equal-width bins) of P(present) raw and Platt (fit on val_merged, 2-fold by image on val itself),
              ECE of box confidence P(present)*P(IoU>=.5) vs "present and IoU>=.5"
  gref_glancex  Glance-X's own definitions (glancex/evaluate.py summarize): acc50, multi_f1 (decoded set, greedy IoU .5,
              mean over multi-target requests, no presence gate), multi_hit_any, multi_count_acc, reject_rate, auroc
  grec        official GREC (gRefCOCO RefExpEvaluator port, scripts/sprint_abs/lib.py): Pr@(F1=1, GIoU>=.5), N-acc, T-acc
              at score threshold 0.7 on grec_val (= val_merged, gRefCOCO val + RefCOCO val singles, as in
              scripts/sprint_abs) / testA / testB (as released).
              Prediction sets: single = top box scored P(present); multi = decoded multi-box set (objectness>.5, NMS .5)
              each scored P(present) (top box if the set is empty); multi_prod = same boxes scored P * objectness;
              *_platt = with the val-fitted Platt presence. Also the best threshold on val and its test transfer.
  diag        Glance-Data diagnostic set by kind (Glance-X definitions: ungated acc50 / reject_rate / multi_hit_any /
              multi_f1) and by kind:lang, plus overall AUROC, multi F1, exact count, Acc@0.5 by language
"""
import argparse, collections, gzip, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sprint_abs"))
from lib import grec_metrics  # noqa: E402

REFCOCO = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocoplus_testA", "refcocoplus_testB",
           "refcocog_val", "refcocog_test"]


def load(fn):
    return [json.loads(l) for l in gzip.open(fn, "rt")]


def sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -60 else 0.0


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


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
    order = np.argsort(allv)
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Platt:
    """Logistic regression on the presence logit (Newton, tiny L2)."""

    def fit(self, z, y, iters=50):
        z, y = np.asarray(z, float), np.asarray(y, float)
        w = np.zeros(2)
        X = np.stack([z, np.ones_like(z)], 1)
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(X @ w, -60, 60)))
            g = X.T @ (p - y) + 1e-4 * w
            H = (X * (p * (1 - p))[:, None]).T @ X + 1e-4 * np.eye(2)
            w -= np.linalg.solve(H, g)
        self.w = w
        return self

    def __call__(self, z):
        return 1 / (1 + np.exp(-np.clip(np.asarray(z, float) * self.w[0] + self.w[1], -60, 60)))


# ------------------------------------------------------------------------------------------- multi-box decoding
def decode_multi(r, thr=0.5, nms=0.5):
    """Glance-X heads.decode_multi on the saved top-20 candidates -> list of (box px clamped, objectness)."""
    W, H = r["W"], r["H"]
    c = [x for x in r["cand"] if x[0] > thr]
    c.sort(key=lambda x: -x[0])
    sel = []
    for x in c:
        if all(iou(x[1:], y[1:]) < nms for y in sel):
            sel.append(x)
    out = []
    for x in sel:
        b = [min(max(v, 0.0), 1.0) for v in x[1:]]
        out.append(([b[0] * W, b[1] * H, b[2] * W, b[3] * H], x[0]))
    return out


def f1_sets(pred, gt, thr=0.5):
    """glancex.evaluate.f1_sets (greedy per prediction, in the given order)."""
    used, tp = set(), 0
    for p in pred:
        best, bj = 0.0, -1
        for j, g in enumerate(gt):
            if j not in used:
                v = iou(p, g)
                if v > best:
                    best, bj = v, j
        if best >= thr:
            used.add(bj); tp += 1
    prec, rec = tp / max(1, len(pred)), tp / max(1, len(gt))
    return 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)


def P(r):
    return sig(r["p_logit"])


# ------------------------------------------------------------------------------------------- sections
def glancex_summary(rows):
    """glancex/evaluate.py summarize(), on our rows."""
    s = dict(n=len(rows))
    single = [r for r in rows if r["present"] and r["n_targets"] == 1]
    multi = [r for r in rows if r["present"] and r["n_targets"] > 1]
    absent = [r for r in rows if not r["present"]]
    pos = single + multi
    if single:
        s["acc50"] = float(np.mean([iou(r["box"], r["gt"][0]) >= 0.5 for r in single]))
    if multi:
        dec = [[b for b, _ in decode_multi(r)] for r in multi]
        s["multi_f1"] = float(np.mean([f1_sets(d, r["gt"]) for d, r in zip(dec, multi)]))
        s["multi_hit_any"] = float(np.mean([max(iou(r["box"], g) for g in r["gt"]) >= 0.5 for r in multi]))
        s["multi_count_acc"] = float(np.mean([len(d) == r["n_targets"] for d, r in zip(dec, multi)]))
    if pos:
        s["present_rate"] = float(np.mean([P(r) >= 0.5 for r in pos]))
    if absent:
        s["reject_rate"] = float(np.mean([P(r) < 0.5 for r in absent]))
    if pos and absent:
        s["auroc"] = auroc([P(r) for r in rows], [int(r["present"]) for r in rows])
    return s


def ours_summary(rows, platt=None, cv_platt=False):
    """eval.py summarize() definitions (+ Platt-calibrated presence)."""
    s = dict(n=len(rows))
    p = np.array([P(r) for r in rows]); y = np.array([int(r["present"]) for r in rows])
    single = [i for i, r in enumerate(rows) if r["present"] and r["n_targets"] == 1]
    multi = [i for i, r in enumerate(rows) if r["present"] and r["n_targets"] > 1]
    absent = [i for i, r in enumerate(rows) if not r["present"]]
    ious = {i: iou(rows[i]["box"], rows[i]["gt"][0]) for i in single}
    if single:
        s["acc50"] = float(np.mean([ious[i] >= 0.5 for i in single]))
        s["T_acc_single"] = float(np.mean([ious[i] >= 0.5 and p[i] >= 0.5 for i in single]))
    if absent:
        s["N_acc"] = float(np.mean(p[absent] < 0.5))
        if multi:
            s["present_recall_multi"] = float(np.mean(p[multi] >= 0.5))
        s["auroc_presence"] = auroc(p, y)
        s["ece_presence"] = ece(p, y)
        z = np.array([r["p_logit"] for r in rows])
        if platt is not None:
            pc = platt(z)
            s["platt"] = dict(w=[float(v) for v in platt.w], ece_presence=ece(pc, y), N_acc=float(np.mean(pc[absent] < 0.5)),
                              T_acc_single=float(np.mean([ious[i] >= 0.5 and pc[i] >= 0.5 for i in single])) if single else None)
        if cv_platt:   # 2-fold by image on this set itself
            img = np.array([r["image_id"] for r in rows])
            fold = img % 2
            pc = np.zeros(len(rows))
            for f in (0, 1):
                m = Platt().fit(z[fold != f], y[fold != f])
                pc[fold == f] = m(z[fold == f])
            s["platt_cv2"] = dict(ece_presence=ece(pc, y), N_acc=float(np.mean(pc[absent] < 0.5)))
    sc = [i for i in range(len(rows)) if i in ious or not rows[i]["present"]]
    pi = np.array([sig(r["iou_logit"]) for r in rows])
    s["ece_box_conf"] = ece([p[i] * pi[i] for i in sc], [int(rows[i]["present"] and ious.get(i, 0) >= 0.5) for i in sc])
    return s


def grec_sets(rows, pres, kind):
    S = []
    for r, p in zip(rows, pres):
        gts = r["gt"] if r["present"] else []
        if kind == "single":
            S.append(([r["box"]], [float(p)], gts))
            continue
        dec = decode_multi(r)
        if not dec:
            S.append(([r["box"]], [float(p)], gts))
        elif kind == "multi":
            S.append(([b for b, _ in dec], [float(p)] * len(dec), gts))
        else:  # multi_prod
            S.append(([b for b, _ in dec], [float(p) * o for _, o in dec], gts))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_dir")
    ap.add_argument("out")
    a = ap.parse_args()
    R = {fn[:-9]: load(os.path.join(a.rows_dir, fn)) for fn in sorted(os.listdir(a.rows_dir)) if fn.endswith(".jsonl.gz")}
    res = dict(rows_dir=a.rows_dir, n_rows={k: len(v) for k, v in R.items()})
    # ---- RefCOCO
    rc = {}
    for s in REFCOCO:
        if s in R:
            rc[s] = float(np.mean([iou(r["box"], r["gt"][0]) >= 0.5 for r in R[s]]))
    if len(rc) == 8:
        rc["mean"] = float(np.mean([rc[s] for s in REFCOCO]))
    res["refcoco"] = rc
    # ---- gRefCOCO
    gsets = [s for s in ("grefcoco_val_merged", "grefcoco_testA", "grefcoco_testB") if s in R]
    platt = None
    if "grefcoco_val_merged" in R:
        v = R["grefcoco_val_merged"]
        platt = Platt().fit([r["p_logit"] for r in v], [int(r["present"]) for r in v])
    res["gref_ours"] = {s: ours_summary(R[s], platt=platt if s != "grefcoco_val_merged" else None,
                                        cv_platt=s == "grefcoco_val_merged") for s in gsets}
    res["gref_glancex"] = {s: glancex_summary(R[s]) for s in gsets}
    # ---- official GREC
    grec_rows = {}
    if "grefcoco_val_merged" in R:
        grec_rows["grec_val"] = R["grefcoco_val_merged"]   # as scripts/sprint_abs (dump.py merged_grefcoco): 25,063
    for sp in ("testA", "testB"):
        if f"grefcoco_{sp}" in R:
            grec_rows[f"grec_{sp}"] = R[f"grefcoco_{sp}"]
    grec = {}
    best = {}
    for name, rows in grec_rows.items():
        raw = [P(r) for r in rows]
        cal = list(platt([r["p_logit"] for r in rows])) if platt is not None else raw
        d = {}
        for kind in ("single", "multi", "multi_prod"):
            for tag, pres in (("raw", raw), ("platt", cal)):
                S = grec_sets(rows, pres, kind)
                d[f"{kind}_{tag}@0.7"] = grec_metrics(S, thresh_score=0.7)
                if name == "grec_val":
                    curve = [(grec_metrics(S, thresh_score=t)["pr_f1"], round(float(t), 2)) for t in np.arange(0.3, 0.96, 0.05)]
                    best[f"{kind}_{tag}"] = max(curve)
                elif f"{kind}_{tag}" in best:
                    t = best[f"{kind}_{tag}"][1]
                    d[f"{kind}_{tag}@val_best({t})"] = grec_metrics(S, thresh_score=t)
        grec[name] = d
    res["grec"] = grec
    res["grec_val_best_threshold"] = {k: dict(pr_f1=v[0], t=v[1]) for k, v in best.items()}
    # ---- Glance-Data diagnostic
    if "diag" in R:
        rows = R["diag"]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["kind"]].append(r); by[f"{r['kind']}:{r['lang']}"].append(r)
        d = dict(n=len(rows), overall=glancex_summary(rows), by_kind={k: glancex_summary(v) for k, v in sorted(by.items())})
        for lang in ("en", "zh"):
            sl = [r for r in rows if r["lang"] == lang and r["present"] and r["n_targets"] == 1]
            d[f"acc50_{lang}"] = float(np.mean([iou(r["box"], r["gt"][0]) >= 0.5 for r in sl])) if sl else None
        res["diag"] = d
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(dict(refcoco=rc, gref={s: {k: res["gref_glancex"][s].get(k) for k in ("reject_rate", "auroc", "multi_f1")}
                                         for s in gsets}), indent=None))


if __name__ == "__main__":
    main()
