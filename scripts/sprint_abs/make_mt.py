"""Multi-target training questions (annotations only; runs anywhere): gRefCOCO *train* multi-target expressions on v1's
own natural training images (never eval or calibration images), written as ONE QUESTION PER TARGET with the same text.
Averaged over the copies, the center soft-CE target becomes the mixture over all referred objects, the edge / GIoU /
IoU-head losses stay per-object, so the single-box head learns a multi-modal center distribution that the suppression-
peak decoder (dump.py `peaks`) turns into a box set. No code change in the trainer.

python scripts/sprint_abs/make_mt.py --ann data/ann --pool data/prep_v1/pool_natural.jsonl --calib data/prep_v1/calib_natural.jsonl \
   --out X/mt.jsonl --n_images 3000"""
import argparse, collections, json, random, re

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True); ap.add_argument("--pool", required=True); ap.add_argument("--calib", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n_images", type=int, default=3000); ap.add_argument("--seed", type=int, default=2)
a = ap.parse_args()
rng = random.Random(a.seed)
pool = {}
for l in open(a.pool):
    r = json.loads(l)
    pool.setdefault(r["image_id"], r) if "image_id" in r else None
calib = {json.loads(l)["image_id"] for l in open(a.calib)}
coco = json.load(open(f"{a.ann}/grefcoco/instances.json"))
ann_box = {an["id"]: [an["bbox"][0], an["bbox"][1], an["bbox"][0] + an["bbox"][2], an["bbox"][1] + an["bbox"][3]] for an in coco["annotations"]}
by = collections.defaultdict(list)
for r in json.load(open(f"{a.ann}/grefcoco/grefs_unc.json")):
    if r["split"] != "train" or r["no_target"] or r["image_id"] in calib or r["image_id"] not in pool:
        continue
    ids = [x for x in r["ann_id"] if x != -1] if isinstance(r["ann_id"], list) else [r["ann_id"]]
    if len(ids) < 2 or len(ids) > 6:
        continue
    for s in r["sentences"]:
        by[r["image_id"]].append((re.sub(r"\s+", " ", s["sent"]).strip(), [ann_box[x] for x in ids]))
imgs = sorted(by); rng.shuffle(imgs)
n_q = n_e = 0
with open(a.out, "w") as f:
    for iid in imgs[: a.n_images]:
        base = pool[iid]
        exprs = by[iid]; rng.shuffle(exprs)
        qs = []
        for t, boxes in exprs[:6]:
            n_e += 1
            for b in boxes:
                qs.append(dict(type="box", text=t, present=True, box=b, src="grefcoco_multi"))
        n_q += len(qs)
        f.write(json.dumps(dict(src="natural", img=base["img"], wh=base["wh"], max_vis=640, plen=base["plen"], image_id=iid,
                                questions=qs[:48])) + "\n")
print(json.dumps(dict(images=min(len(imgs), a.n_images), expressions=n_e, questions=n_q, available_images=len(imgs))))
