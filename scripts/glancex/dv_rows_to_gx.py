"""Convert eval.py (v0 path) rows into the dump_rows.py schema, so scripts/glancex/score.py scores both model families.

  python scripts/glancex/dv_rows_to_gx.py --rows ROWS_DIR --prep_dir PREP --out OUT_DIR

eval.py writes one row per request in dataset order (records in file order, requests in record order) without ids or
target lists; this aligns each row with its request in PREP/eval_<set>.jsonl (text must match) and adds image_id, qi,
kind, lang, gt. A single-box model has no objectness head: `cand` holds the single box with objectness 1, so the
"decoded set" is the single box, ungated (as Glance-X's multi_f1 does not gate either). Set gxdiag is renamed diag.
"""
import argparse, gzip, json, math, os


def logit(p, eps=1e-7):
    p = min(max(float(p), eps), 1 - eps)
    return math.log(p / (1 - p))


ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--prep_dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for fn in sorted(os.listdir(a.rows)):
    if not fn.endswith(".jsonl.gz"):
        continue
    name = fn[:-9]
    rows = [json.loads(l) for l in gzip.open(os.path.join(a.rows, fn), "rt")]
    reqs = []
    for l in open(f"{a.prep_dir}/eval_{name}.jsonl"):
        r = json.loads(l)
        for qi, q in enumerate(r["questions"]):
            reqs.append((r, qi, q))
    assert len(reqs) == len(rows), (name, len(reqs), len(rows))
    out_name = "diag" if name == "gxdiag" else name
    with gzip.open(os.path.join(a.out, out_name + ".jsonl.gz"), "wt") as f:
        for row, (r, qi, q) in zip(rows, reqs):
            assert row["text"] == q["text"], (name, row["text"], q["text"])
            W, H = r["W"], r["H"]
            gt = (q.get("boxes") or ([q["box"]] if q.get("box") is not None else [])) if q["present"] else []
            b = row["box"]
            nb = [min(max(b[0] / W, 0), 1), min(max(b[1] / H, 0), 1), min(max(b[2] / W, 0), 1), min(max(b[3] / H, 0), 1)]
            f.write(json.dumps(dict(set=out_name, image_id=r["image_id"], qi=qi, text=q["text"], kind=q.get("kind", "orig"),
                                    lang=q.get("lang", "en"), src=q.get("src"), sent_id=q.get("sent_id"),
                                    present=bool(q["present"]) and len(gt) > 0, n_targets=len(gt), W=W, H=H, gt=gt,
                                    p_logit=logit(row["p_present"]), iou_logit=logit(row["p_iou50"]),
                                    center_top1=row.get("center_p"), box=b, cand=[[1.0] + [round(v, 4) for v in nb]]),
                               ensure_ascii=False) + "\n")
    print(name, len(rows))
