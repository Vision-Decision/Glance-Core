"""Full GREC (Pr@F1=1, N-acc) curve for Glance v1 by sweeping the presence threshold.

Official GREC rule (gRefCOCO code): a predicted box survives iff its score >= t; greedy one-to-one matching at
GIoU >= 0.5; a sample succeeds iff F1 = 1; a no-target sample succeeds iff no box survives. Glance outputs one box
with score sigmoid(p_logit). Run on rows saved by the sprint_abs eval (HF Vision-Decision/glance-train-ckpts-legacy
sprint_abs/rows_v1/grec_*.jsonl.gz), e.g. on gen-host:  python3 grec_curve.py rows_dir out.json
"""
import gzip, json, math, sys

def giou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    cx1, cy1, cx2, cy2 = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
    c = (cx2 - cx1) * (cy2 - cy1)
    iou = inter / ua if ua > 0 else 0.0
    return iou - (c - ua) / c if c > 0 else iou

rows_dir, out = sys.argv[1], sys.argv[2]
res = {}
for split in ("grec_val", "grec_testA", "grec_testB"):
    S = []
    for line in gzip.open(f"{rows_dir}/{split}.jsonl.gz", "rt"):
        r = json.loads(line)
        g = 1 / (1 + math.exp(-r["p_logit"]))
        gts = r.get("gt_boxes") or ([r["gt_box"]] if r.get("gt_box") else [])
        if not r["present"] or not gts:
            S.append((g, 0, False))
        elif len(gts) == 1:
            S.append((g, 1, giou(r["box"], gts[0]) >= 0.5))
        else:
            S.append((g, len(gts), False))
    n, n0 = len(S), sum(1 for s in S if s[1] == 0)
    curve = []
    for k in range(0, 101):
        t = k / 100
        succ_abs = sum(1 for g, nt, ok in S if nt == 0 and g < t)
        succ_tgt = sum(1 for g, nt, ok in S if nt == 1 and g >= t and ok)
        curve.append(dict(t=t, Pr=100 * (succ_abs + succ_tgt) / n, N_acc=100 * succ_abs / max(1, n0)))
    best = max(curve, key=lambda c: c["Pr"])
    res[split] = dict(n=n, n_no_target=n0, n_single=sum(1 for s in S if s[1] == 1), n_multi=sum(1 for s in S if s[1] > 1),
                      at_0_5=next(c for c in curve if abs(c["t"] - .5) < 1e-9),
                      at_0_7=next(c for c in curve if abs(c["t"] - .7) < 1e-9), best_Pr=best, curve=curve)
json.dump(res, open(out, "w"), indent=1)
for s, v in res.items():
    print(s, v["n"], v["n_no_target"], v["n_single"], v["n_multi"], "t=.5", v["at_0_5"], "t=.7", v["at_0_7"], "best", v["best_Pr"])
