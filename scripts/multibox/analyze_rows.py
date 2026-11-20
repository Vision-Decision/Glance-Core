"""Offline analysis of multi-box row dumps (eval.py --rows_dir); no model needed.

python scripts/multibox/analyze_rows.py --calib calibmb.jsonl.gz --sets grec_val.jsonl.gz,cocoall_val.jsonl.gz

  ceiling : F1 if the best subset of the K candidates were chosen (one-to-one, IoU>=0.5) -> geometry vs selection
  thr     : P(match) threshold chosen on the calibration split only (max multi+single F1), then applied to the eval sets
  count   : decode the top-n candidates by P(match), n = argmax of the count head (n>=4 -> all with P(match)>=thr)
"""
import argparse, collections, gzip, json, os
import numpy as np

code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval.py")).read()
ns = dict(np=np, collections=collections)
exec(code[code.index("def ece("):code.index("@torch.no_grad()")], ns)
exec(code[code.index("def _iou("):code.index("def summarize_grec")], ns)
summarize_multi, decode_row, set_tp, nms_list = ns["summarize_multi"], ns["decode_row"], ns["set_tp"], ns["nms_list"]


def load(p):
    return [json.loads(l) for l in gzip.open(p, "rt") if '"kind": "box"' in l or '"kind":"box"' in l]


def gt(r):
    return (r.get("gt_boxes") or ([r["gt_box"]] if r.get("gt_box") else [])) if r["present"] else []


def f1(P, G):
    if not P and not G:
        return 1.0
    return 2 * set_tp(P, G) / (len(P) + len(G))


def by_group(rows, pred):
    out = collections.defaultdict(list)
    for r in rows:
        G = gt(r)
        if r["present"] and not G:
            continue
        k = "none" if not G else ("single" if len(G) == 1 else "multi")
        out[k].append(f1(pred(r), G))
    return {k: round(100 * float(np.mean(v)), 1) for k, v in out.items()} | {"n": {k: len(v) for k, v in out.items()}}


def ceiling(r):
    if r["p_present"] < 0.5:
        return []
    G = gt(r)
    C = [c[:4] for c in r.get("cands") or []] or [r["box"]]
    if not G:
        return [r["box"]]
    # best subset: the candidates matched in a max one-to-one assignment (greedy on IoU) -> F1 = 2m/(m+|G|)
    pairs = sorted(((ns["_iou"](c, g), i, j) for i, c in enumerate(C) for j, g in enumerate(G)), reverse=True)
    ui, uj, keep = set(), set(), []
    for v, i, j in pairs:
        if v >= 0.5 and i not in ui and j not in uj:
            ui.add(i); uj.add(j); keep.append(C[i])
    return keep or [r["box"]]


def count_decode(r, thr):
    if r["p_present"] < 0.5 or not r.get("cands"):
        return decode_row(r, thr)[0]
    n = int(np.argmax(r["count_p"])) + 1
    c = sorted(r["cands"], key=lambda x: -x[5])
    k = nms_list([x[:4] for x in c], [x[5] for x in c], 0.7)
    c = [c[i] for i in k]
    if n >= len(r["count_p"]):
        sel = [x for x in c if x[5] >= thr]
        sel = sel if len(sel) >= n else c[:n]
    else:
        sel = c[:n]
    return [x[:4] for x in sel] or [r["box"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sets", required=True)
    a = ap.parse_args()
    cal = load(a.calib)
    grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    sc = {}
    for t in grid:
        g = by_group(cal, lambda r: decode_row(r, t)[0])
        n = g["n"]
        sc[t] = (g.get("multi", 0) * n.get("multi", 0) + g.get("single", 0) * n.get("single", 0)) / max(1, n.get("multi", 0) + n.get("single", 0))
    best = max(sc, key=sc.get)
    print("calib: F1(multi+single) by thr", {t: round(v, 1) for t, v in sc.items()}, "-> thr", best)
    print("calib: count-guided", by_group(cal, lambda r: count_decode(r, best)), "thr-decode", by_group(cal, lambda r: decode_row(r, best)[0]),
          "as decoded", by_group(cal, lambda r: decode_row(r)[0]))
    res = {}
    for p in a.sets.split(","):
        rows = load(p)
        name = os.path.basename(p).split(".")[0]
        res[name] = dict(
            as_decoded=by_group(rows, lambda r: decode_row(r)[0]),
            thr_calib=by_group(rows, lambda r: decode_row(r, best)[0]),
            count_guided=by_group(rows, lambda r: count_decode(r, best)),
            ceiling=by_group(rows, ceiling),
            full_thr_calib=summarize_multi(rows, thr=best))
        for k in ("as_decoded", "thr_calib", "count_guided", "ceiling"):
            print(name, k, res[name][k])
    print(json.dumps(dict(thr=best, calib_scores=sc, sets=res), default=str)[:200])
    return res


if __name__ == "__main__":
    main()
