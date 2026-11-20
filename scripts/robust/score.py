"""Score glance-robust-ref rows. Per type (and sub-type / language):
  Acc@0.5        present items: IoU>=0.5 and predicted present (P(present)>=0.5); also the ungated box-only accuracy
  false_absent   present items predicted absent
  overconf_err   present items with a wrong box (IoU<0.5) reported with confidence>0.8 (confidence = P(present)*P(IoU>=.5),
                 the released model's `confidence`); predicted-absent items are not counted here
  N_acc          absent items predicted absent;  absent_overconf: absent items with confidence>0.8
Rows: eval.py --rows_dir output (v1 / v1.1: rows carry meta; v0: joined by file order with the prep file), or base_gen.py.
Usage: python score.py --prep /content/robust --rows name=dir[:T_presence] ... --out scores.json
"""
import argparse, collections, gzip, json, math

ap = argparse.ArgumentParser()
ap.add_argument("--prep", default="/content/robust")
ap.add_argument("--sets", default="zh,para,func,vague")
ap.add_argument("--rows", nargs="+", required=True, help="model=rows_dir[:presence_T]")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def load(model_dir, T, name):
    rows = [json.loads(l) for l in gzip.open(f"{model_dir}/robust_{name}.jsonl.gz", "rt")]
    if rows and "meta" not in rows[0]:  # v0 rows: file order
        qs = [q for l in open(f"{a.prep}/eval_robust_{name}.jsonl") for q in json.loads(l)["questions"]]
        assert len(qs) == len(rows)
        for r, q in zip(rows, qs):
            assert r["text"] == q["text"][: len(r["text"])]
            r["meta"] = q["meta"]
    for r in rows:
        if "out" in r:  # generative baseline: hard decision, no confidence
            r["pred"] = r["pred_present"]; r["conf"] = None
        else:
            p = r["p_present"]
            if T and "p_logit" in r:
                p = 1 / (1 + math.exp(-r["p_logit"] / T))
            r["pred"] = p >= 0.5; r["conf"] = p * r.get("p_iou50", 1.0)
    return rows


def metrics(rows):
    pres = [r for r in rows if r["present"]]; ab = [r for r in rows if not r["present"]]
    m = dict(n_present=len(pres), n_absent=len(ab))
    if pres:
        hit = lambda r: r.get("iou", 0) >= 0.5
        m["acc50"] = sum(hit(r) and r["pred"] for r in pres) / len(pres)
        m["acc50_box_only"] = sum(hit(r) for r in pres) / len(pres)
        m["false_absent"] = sum(not r["pred"] for r in pres) / len(pres)
        if pres[0]["conf"] is not None:
            m["overconf_err"] = sum(r["pred"] and not hit(r) and r["conf"] > 0.8 for r in pres) / len(pres)
    if ab:
        m["N_acc"] = sum(not r["pred"] for r in ab) / len(ab)
        if ab[0]["conf"] is not None:
            m["absent_overconf"] = sum(r["pred"] and r["conf"] > 0.8 for r in ab) / len(ab)
    return m


res = {}
for spec in a.rows:
    name, rest = spec.split("=", 1)
    d, _, T = rest.partition(":")
    T = float(T) if T else None
    res[name] = dict(presence_T=T, sets={})
    for s in a.sets.split(","):
        rows = load(d, T, s)
        e = dict(all=metrics(rows))
        for key in ("sub", "lang"):
            g = collections.defaultdict(list)
            for r in rows:
                g[r["meta"][key]].append(r)
            e[key] = {k: metrics(v) for k, v in sorted(g.items())}
        res[name]["sets"][s] = e
json.dump(res, open(a.out, "w"), indent=1)
hdr = f"{'model':10s} {'set':6s} {'Acc@.5':>7s} {'box':>6s} {'falseAbs':>8s} {'overconf':>8s} {'N-acc':>6s} {'abs>0.8':>7s}"
print(hdr)
f = lambda v: "   -  " if v is None else f"{100 * v:6.1f}"
for name, r in res.items():
    for s, e in r["sets"].items():
        m = e["all"]
        print(f"{name:10s} {s:6s} {f(m.get('acc50'))} {f(m.get('acc50_box_only'))} {f(m.get('false_absent'))}   "
              f"{f(m.get('overconf_err'))} {f(m.get('N_acc'))} {f(m.get('absent_overconf'))}")
