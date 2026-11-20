"""Evaluate a Glance checkpoint on the Glance-Data diagnostic split (+ the pos_pos_sole add-on), per kind x lang.

  cd /content/dv && python scripts/ground_v2/eval_diag.py --ckpt /content/ckpt/v1/ckpt_final.pt \
      --recs /content/gd/diagnostic.jsonl,/content/gd/diagnostic_pos_sole.jsonl --out /content/diag/v1.json --rows /content/diag/v1_rows.jsonl.gz

Metrics (threshold 0.5 on raw P(present); no temperature):
  1-target questions   acc50 = P>=0.5 and IoU(pred, target)>=0.5 ("gated"); also acc50_box (IoU only) and recall (P>=0.5)
  multi-target (n>=2)  recall = P>=0.5; hit_any = P>=0.5 and IoU with any target >=0.5; union50 = P>=0.5 and IoU with the
                       union box >=0.5 (the single-box head can only output one box)
  absent (n=0)         n_acc = P<0.5
Keys: "<qtype>/<lang>" and "<qtype>/all"; `function` is split into function@1 / function@multi by its target count.
Also summarize offline from saved rows: python eval_diag.py --summarize rows.jsonl.gz --out x.json
"""
import argparse, collections, gzip, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def summarize(rows, thr=0.5):
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows:
        m = r["meta"]; qt, lang, n = m["qtype"], m["lang"], int(r.get("n_targets") or 0)
        if qt == "function":
            qt = "function@1" if n == 1 else "function@multi"
        p = r["p_present"]; gts = r.get("gt_boxes") or []
        for key in (f"{qt}/{lang}", f"{qt}/all"):
            s = acc[key]; s["n"] += 1
            if n == 0:
                s["n_acc"] += p < thr
            elif n == 1:
                v = iou(r["box"], gts[0])
                s["acc50"] += p >= thr and v >= 0.5; s["acc50_box"] += v >= 0.5; s["recall"] += p >= thr
            else:
                best = max(iou(r["box"], g) for g in gts)
                u = [min(g[0] for g in gts), min(g[1] for g in gts), max(g[2] for g in gts), max(g[3] for g in gts)]
                s["recall"] += p >= thr; s["hit_any"] += p >= thr and best >= 0.5; s["union50"] += p >= thr and iou(r["box"], u) >= 0.5
    out = {}
    for k, s in sorted(acc.items()):
        n = s.pop("n")
        out[k] = dict(n=int(n), **{m: round(100 * v / n, 1) for m, v in s.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--recs", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="")
    ap.add_argument("--summarize", default="", help="rows.jsonl.gz to summarize without a GPU")
    a = ap.parse_args()
    if a.summarize:
        rows = [json.loads(l) for l in gzip.open(a.summarize, "rt")]
    else:
        import torch
        from dv.config import Config
        from dv.model.build import build, load_ckpt
        from eval import predict_v1
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = Config().update(sd["cfg"])
        net, heads, proc = build(cfg, "cuda")
        load_ckpt(a.ckpt, net, heads)
        net.cfg_proc = proc
        net.eval(); heads.eval()
        del sd
        recs = []
        for p in a.recs.split(","):
            for l in open(p):
                r = json.loads(l)
                for q in r["questions"]:
                    q["meta"] = dict(qtype=q["qtype"], lang=q["lang"], src=q["src"])
                recs.append(r)
        rows = predict_v1(net, heads, recs, cfg.data_root, cfg)
        if a.rows:
            with gzip.open(a.rows, "wt") as f:
                for r in rows:
                    f.write(json.dumps({k: r.get(k) for k in ("text", "present", "p_present", "p_iou50", "box", "gt_boxes",
                                                             "n_targets", "meta")}, ensure_ascii=False) + "\n")
    res = dict(ckpt=a.ckpt, n_rows=len(rows), per_kind=summarize(rows))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    for k, v in res["per_kind"].items():
        if k.endswith("/all"):
            print(k, v, flush=True)


if __name__ == "__main__":
    main()
