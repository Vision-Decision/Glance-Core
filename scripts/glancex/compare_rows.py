"""Equivalence of two row dumps (dump_rows.py --impl glancex vs --impl dv) of the same checkpoint, sets and batching.

  python scripts/glancex/compare_rows.py DIR_A DIR_B OUT.json
Per set: rows matched on (image_id, qi, text); max |diff| of presence / IoU logits, box (px) and candidate tensors,
and whether every derived decision (P>=.5, Acc@0.5 hit, decoded multi-box count) agrees.
"""
import gzip, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import decode_multi, iou  # noqa: E402

A, B, out = sys.argv[1:4]
res = {}
for fn in sorted(os.listdir(A)):
    if not fn.endswith(".jsonl.gz") or not os.path.exists(os.path.join(B, fn)):
        continue
    ra = {(r["image_id"], r["qi"], r["text"]): r for r in map(json.loads, gzip.open(os.path.join(A, fn), "rt"))}
    rb = {(r["image_id"], r["qi"], r["text"]): r for r in map(json.loads, gzip.open(os.path.join(B, fn), "rt"))}
    keys = sorted(set(ra) & set(rb), key=str)
    d = dict(n_a=len(ra), n_b=len(rb), n_matched=len(keys), max_abs_p_logit=0.0, max_abs_iou_logit=0.0, max_abs_box_px=0.0,
             max_abs_cand=0.0, decision_mismatch=0)
    for k in keys:
        x, y = ra[k], rb[k]
        d["max_abs_p_logit"] = max(d["max_abs_p_logit"], abs(x["p_logit"] - y["p_logit"]))
        d["max_abs_iou_logit"] = max(d["max_abs_iou_logit"], abs(x["iou_logit"] - y["iou_logit"]))
        d["max_abs_box_px"] = max(d["max_abs_box_px"], max(abs(u - v) for u, v in zip(x["box"], y["box"])))
        d["max_abs_cand"] = max(d["max_abs_cand"], max(abs(u - v) for cu, cv in zip(x["cand"], y["cand"]) for u, v in zip(cu, cv)))
        hit = lambda r: bool(r["gt"]) and iou(r["box"], r["gt"][0]) >= 0.5
        if ((x["p_logit"] >= 0) != (y["p_logit"] >= 0)) or hit(x) != hit(y) or len(decode_multi(x)) != len(decode_multi(y)):
            d["decision_mismatch"] += 1
    d["bitwise_equal"] = all(d[k] == 0 for k in ("max_abs_p_logit", "max_abs_iou_logit", "max_abs_box_px", "max_abs_cand"))
    res[fn[:-9]] = d
    print(fn, json.dumps(d))
json.dump(res, open(out, "w"), indent=1)
