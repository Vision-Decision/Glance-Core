"""gRefCOCO *train* expressions (no-, single-, multi-target) on the 600 held-out calibration images of calib_natural
(images excluded from v1 training). Used only to tune multi-target GREC decoding and GREC thresholds; never eval data.
python scripts/sprint_abs/make_grefcalib.py --ann data/ann --calib data/prep_v1/calib_natural.jsonl --out X/grefcalib.jsonl"""
import argparse, collections, json, re

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True); ap.add_argument("--calib", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
calib = {}
for l in open(a.calib):
    r = json.loads(l); calib[r["image_id"]] = r
coco = json.load(open(f"{a.ann}/grefcoco/instances.json"))
ann_box = {an["id"]: [an["bbox"][0], an["bbox"][1], an["bbox"][0] + an["bbox"][2], an["bbox"][1] + an["bbox"][3]]
           for an in coco["annotations"]}
grefs = json.load(open(f"{a.ann}/grefcoco/grefs_unc.json"))
by = collections.defaultdict(list); c = collections.Counter(); seen = set()
for r in grefs:
    if r["image_id"] not in calib:
        continue
    assert r["split"] == "train", r["split"]
    ids = [x for x in r["ann_id"] if x != -1] if isinstance(r["ann_id"], list) else [r["ann_id"]]
    no_t = bool(r["no_target"])
    boxes = [] if no_t else [ann_box[x] for x in ids]
    for s in r["sentences"]:
        t = re.sub(r"\s+", " ", s["sent"]).strip()
        if (r["image_id"], t.lower()) in seen:
            continue
        seen.add((r["image_id"], t.lower()))
        by[r["image_id"]].append(dict(type="box", text=t, present=not no_t, box=boxes[0] if len(boxes) == 1 else None,
                                      boxes=boxes, n_targets=len(boxes), src="grefcoco_train"))
        c[min(len(boxes), 3)] += 1
with open(a.out, "w") as f:
    for iid, qs in by.items():
        base = calib[iid]
        for i in range(0, len(qs), 48):
            f.write(json.dumps(dict(src="natural", img=base["img"], wh=base["wh"], max_vis=640, plen=base["plen"],
                                    image_id=iid, questions=qs[i:i + 48])) + "\n")
print(len(by), "images", dict(c))
