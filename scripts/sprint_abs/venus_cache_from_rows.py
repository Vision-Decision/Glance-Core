"""Build the VenusBench wrapper cache (venus_glance.py GLANCE_CACHE) from dump.py rows of the same checkpoint, so the
official evaluator is served the identical forward without a second pass.
python scripts/sprint_abs/venus_cache_from_rows.py ROWS.jsonl.gz ANNS_DIR IMAGES_DIR OUT.json"""
import gzip, json, os, sys

rows, anns, images, out = sys.argv[1:5]
tab = {}
for l in gzip.open(rows, "rt"):
    r = json.loads(l)
    tab[(r["img"], r["text"])] = dict(p_logit=r["p_logit"], iou_logit=r["iou_logit"], box=r["box"], wh=r["wh"],
                                      center_top1=r["center_top1"], center_ent=r["center_ent"], edge_sd=r["edge_sd"])
need, miss = 0, 0
for fn in sorted(os.listdir(anns)):
    if fn.endswith(".json"):
        for it in json.load(open(os.path.join(anns, fn))):
            need += 1
            miss += (os.path.join(images, it["img_filename"]), it["instruction"]) not in tab
print("items", need, "missing", miss, "table", len(tab))
assert miss == 0
json.dump({json.dumps(list(k)): v for k, v in tab.items()}, open(out, "w"))
