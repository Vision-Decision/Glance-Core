"""Gen-v1 rows -> cache for the OFFICIAL VenusBench-GD evaluator wrapper (scripts/sprint_abs/venus_glance.py, GLANCE_CACHE),
so UI-Venus eval.py (unmodified) scores Gen-v1's decoded boxes; with GLANCE_GATED=1 a refusal returns no point.
python gen_v1/venus_cache.py ROWS.jsonl.gz ANNS_DIR IMAGES_DIR OUT.json {greedy|pnone}
  greedy: refuse iff the first answer token's argmax is "none" (the generative fixed point)
  pnone : refuse iff q(none) > 0.5 (the presence dial at 0.5)
The wrapper refuses iff sigmoid(p_logit) < 0.5, so p_logit encodes the rule; box = grammar-constrained greedy decode (px)."""
import gzip, json, math, os, sys

rows, anns, images, out, mode = sys.argv[1:6]
tab = {}
for l in gzip.open(rows, "rt"):
    r = json.loads(l)
    if mode == "greedy":
        pl = -10.0 if r["greedy_none"] else 10.0
    else:
        p = min(max(1 - r["p_none"], 1e-6), 1 - 1e-6); pl = math.log(p / (1 - p))
    tab[(r["img"], r["text"])] = dict(p_logit=pl, iou_logit=0.0, box=r["box_px"], wh=r["wh"], center_top1=0.0,
                                      center_ent=0.0, edge_sd=[0.0] * 4)
need = miss = 0
for fn in sorted(os.listdir(anns)):
    if fn.endswith(".json"):
        for it in json.load(open(os.path.join(anns, fn))):
            need += 1
            miss += (os.path.join(images, it["img_filename"]), it["instruction"]) not in tab
print("items", need, "missing", miss, "table", len(tab))
assert miss == 0
json.dump({json.dumps(list(k)): v for k, v in tab.items()}, open(out, "w"))
