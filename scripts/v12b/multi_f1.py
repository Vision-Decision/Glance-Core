"""gRefCOCO multi-target set F1 with Glance-X's definition (reports/Glance-X 评估.md §2): only expressions with >= 2
referents; predicted set vs GT set, greedy one-to-one matching at IoU >= 0.5, F1 = 2TP / (|P| + |G|) per expression,
averaged. Glance-X does not gate the set by P(present), so `ungated` is the comparable number; `gated` is ours.

  python scripts/v12b/multi_f1.py --rows rows/grec_val.jsonl.gz --thr 0.5 --mode thr|count [--calib rows/calibmb.jsonl.gz]
With --calib, the decode threshold is also chosen on the calibration rows (max mean F1 over all calib questions with
referents) and the eval set is scored at that threshold.
"""
import argparse, collections, gzip, json, os
import numpy as np

code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval.py")).read()
ns = dict(np=np, collections=collections)
exec(code[code.index("def _iou("):code.index("def summarize_grec")], ns)
decode_row, set_tp = ns["decode_row"], ns["set_tp"]


def load(p):
    return [json.loads(l) for l in gzip.open(p, "rt") if '"kind": "box"' in l]


def gt(r):
    return (r.get("gt_boxes") or ([r["gt_box"]] if r.get("gt_box") else [])) if r["present"] else []


def f1(P, G):
    return 1.0 if not P and not G else 2 * set_tp(P, G) / (len(P) + len(G))


def score(rows, thr, mode, gated, min_g=2):
    v = []
    for r in rows:
        G = gt(r)
        if len(G) < min_g:
            continue
        rr = r if gated else dict(r, p_present=1.0)
        v.append(f1(decode_row(rr, thr, mode=mode)[0], G))
    return 100 * float(np.mean(v)) if v else None, len(v)


def selection(rows, thr, mode, min_g=2):
    """how the arm selects on multi-referent rows: set sizes, exact-count rate, precision / recall of the selected boxes,
    and how many correct candidates were available (oracle recall over all candidates)"""
    nP = nG = tp = exact = orc = 0; n = 0
    for r in rows:
        G = gt(r)
        if len(G) < min_g:
            continue
        P = decode_row(dict(r, p_present=1.0), thr, mode=mode)[0]
        t = set_tp(P, G); n += 1; nP += len(P); nG += len(G); tp += t; exact += len(P) == len(G)
        C = [c[:4] if isinstance(c, (list, tuple)) and len(c) >= 4 else c for c in (r.get("cands") or [])]
        try:
            orc += set_tp([c[0] if isinstance(c[0], (list, tuple)) else c[:4] for c in r.get("cands") or []], G)
        except Exception:
            pass
    return dict(n=n, mean_pred=round(nP / max(1, n), 2), mean_gt=round(nG / max(1, n), 2), exact_count=round(exact / max(1, n), 3),
                precision=round(tp / max(1, nP), 3), recall=round(tp / max(1, nG), 3), oracle_recall=round(orc / max(1, nG), 3))


ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--mode", default="thr")
ap.add_argument("--calib", default="")
a = ap.parse_args()
rows = load(a.rows)
out = {"thr": a.thr, "mode": a.mode}
for g in (False, True):
    out["gated" if g else "ungated"] = score(rows, a.thr, a.mode, g)
if a.calib:
    cal = load(a.calib)
    best = max((score(cal, t, a.mode, True, min_g=1)[0] or 0, t) for t in np.arange(0.1, 0.95, 0.05))
    out["calib_thr"] = round(float(best[1]), 2)
    out["calib_ungated"] = score(rows, best[1], a.mode, False)
    out["calib_gated"] = score(rows, best[1], a.mode, True)
out["selection"] = selection(rows, out.get("calib_thr", a.thr), a.mode)
print(json.dumps(out))
