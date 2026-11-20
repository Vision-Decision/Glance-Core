"""Pilot data for the v1.2 multi-box head, from annotations only (runs locally; no images are read).

python scripts/multibox/prep_pilot.py --ann ../../data/ann --pool ../../data/prep_v1 --prep data/prep --out data/prep_mb

Sources (exact boxes, rules only; no generated text):
  G  gRefCOCO train multi-target expressions (dropped by dv/data/prep.py until v1.2), boxes = their COCO instances.
  C  "category = all instances" questions built from COCO instances_train2014: for a category with n>=1 instances in
     the image, one phrasing out of {"<plural>", "all <plural>", "all the <plural>", "every <name>", "<name>"} asks for
     all n boxes (n=1 is a plain single target). Skipped: categories with any iscrowd instance, any instance smaller than
     --min_side px on a side (invisible at 32 px patches; keeping them would teach wrong counts), or more than --max_n
     instances. Absent: categories with no instance in the image (COCO labels its 80 categories exhaustively).
  R  replay of the v1.1 natural-image pool (RefCOCO/+/g single targets + absent / hard negatives), so that single-target
     behaviour and the presence gate see their usual data.
Exclusions: every image of every eval split in --prep (RefCOCO/+/g val/test*, gRefCOCO val/testA/testB) and the v1.1
calibration images (calib_natural.jsonl) are never used for training.

Outputs (--out): train_mb.jsonl (v1 records), calib_mb.jsonl (G + C questions on the v1.1 calibration images, for
choosing decode thresholds; never eval images), stats.json; and eval_cocoall_val.jsonl (--prep, eval format: C on the
gRefCOCO val images, which are held out of training) for a "bare category = all instances" eval set.
"""
import argparse, collections, glob, json, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dv.data.prep_v1 import plen_image  # noqa: E402  (stdlib-only module)

PLURAL = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "skis", "scissors": "scissors",
          "bus": "buses", "couch": "couches", "bench": "benches", "sandwich": "sandwiches", "knife": "knives",
          "wine glass": "wine glasses", "glass": "glasses", "dining table": "dining tables", "toothbrush": "toothbrushes",
          "hair drier": "hair driers", "skateboard": "skateboards", "potted plant": "potted plants", "giraffe": "giraffes",
          "broccoli": "broccoli", "teddy bear": "teddy bears", "sports ball": "balls", "tie": "ties"}
NAME = {"sports ball": "ball", "dining table": "table", "potted plant": "potted plant", "hair drier": "hair dryer",
        "cell phone": "cell phone", "tv": "tv", "motorcycle": "motorcycle"}
TEMPLATES = ["{pl}", "all {pl}", "all the {pl}", "every {nm}", "{nm}"]


def plural(c):
    if c in PLURAL:
        return PLURAL[c]
    if c.endswith(("s", "x", "ch", "sh")):
        return c + "es"
    if c.endswith("y") and c[-2] not in "aeiou":
        return c[:-1] + "ies"
    return c + "s"


def norm_text(s):
    return re.sub(r"\s+", " ", s.strip()).rstrip(".").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/ann")
    ap.add_argument("--pool", default="data/prep_v1")
    ap.add_argument("--prep", default="data/prep")
    ap.add_argument("--out", default="data/prep_mb")
    ap.add_argument("--n_multi", type=int, default=40000, help="gRefCOCO multi-target expressions (max)")
    ap.add_argument("--n_cat_images", type=int, default=20000, help="images that get category-all questions")
    ap.add_argument("--n_replay", type=int, default=45000, help="replayed v1.1 natural questions")
    ap.add_argument("--absent_per_image", type=int, default=1)
    ap.add_argument("--min_side", type=float, default=16)
    ap.add_argument("--max_n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    coco = json.load(open(f"{a.ann}/annotations/instances_train2014.json"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_info = {im["id"]: im for im in coco["images"]}
    inst = collections.defaultdict(lambda: collections.defaultdict(list))   # image -> cat -> [(box, crowd)]
    ann_box = {}
    for an in coco["annotations"]:
        x, y, w, h = an["bbox"]
        b = [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]
        inst[an["image_id"]][an["category_id"]].append((b, an["iscrowd"], w, h))
        ann_box[an["id"]] = b

    held = set()
    for fn in glob.glob(f"{a.prep}/eval_*.jsonl"):
        if "cocoall" in fn:
            continue
        for l in open(fn):
            held.add(json.loads(l)["image_id"])
    calib = {json.loads(l)["image_id"] for l in open(f"{a.pool}/calib_natural.jsonl")}
    print(f"held-out eval images {len(held)}, v1.1 calibration images {len(calib)}")

    # ---- G: gRefCOCO multi-target train expressions
    grefs = json.load(open(f"{a.ann}/grefcoco/grefs_unc.json"))
    G = collections.defaultdict(list)
    n_g_all = 0
    for r in grefs:
        if r["split"] != "train" or r["no_target"]:
            continue
        ids = [x for x in r["ann_id"] if x != -1] if isinstance(r["ann_id"], list) else [r["ann_id"]]
        if len(ids) < 2:
            continue
        iid = r["image_id"]
        if iid in held:
            continue
        boxes = [ann_box[x] for x in ids]
        for s in r["sentences"]:
            n_g_all += 1
            G[iid].append(dict(type="box", text=norm_text(s["sent"]), present=True, box=None, boxes=boxes,
                               n_targets=len(boxes), src="grefcoco_multi"))
    # ---- C: category-all questions
    def cat_questions(iid, rng, absent_n):
        qs = []
        im = inst.get(iid, {})
        for cid, lst in sorted(im.items()):
            if any(c for _, c, _, _ in lst) or len(lst) > a.max_n or any(min(w, h) < a.min_side for _, _, w, h in lst):
                continue
            nm = NAME.get(cats[cid], cats[cid])
            t = rng.choice(TEMPLATES).format(pl=plural(nm), nm=nm)
            boxes = [b for b, _, _, _ in lst]
            qs.append(dict(type="box", text=t, present=True, box=boxes[0] if len(boxes) == 1 else None, boxes=boxes,
                           n_targets=len(boxes), src="cat_all"))
        absent = [c for c in cats if c not in im]
        for cid in rng.sample(absent, min(absent_n, len(absent))):
            nm = NAME.get(cats[cid], cats[cid])
            t = rng.choice(TEMPLATES).format(pl=plural(nm), nm=nm)
            qs.append(dict(type="box", text=t, present=False, box=None, boxes=[], n_targets=0, src="cat_all_absent",
                           neg="easy"))
        return qs

    # ---- R: replay pool (v1.1 natural)
    pool = {}
    for l in open(f"{a.pool}/pool_natural.jsonl"):
        r = json.loads(l)
        pool.setdefault(r["image_id"], dict(r, questions=[]))["questions"] += r["questions"]
    train_imgs = [i for i in sorted(set(pool) | set(G)) if i not in held and i not in calib]
    rng.shuffle(train_imgs)

    g_keys = [(iid, k) for iid in train_imgs for k in range(len(G.get(iid, [])))]
    rng.shuffle(g_keys)
    g_keep = set(g_keys[: a.n_multi])
    cat_imgs = set(train_imgs[: a.n_cat_images])
    rep = [(iid, k) for iid in train_imgs if iid in pool for k in range(len(pool[iid]["questions"]))]
    rng.shuffle(rep)
    rep_keep = set(rep[: a.n_replay])

    recs, cnt = [], collections.Counter()
    for iid in train_imgs:
        qs = [q for k, q in enumerate(G.get(iid, [])) if (iid, k) in g_keep]
        if iid in cat_imgs:
            qs += cat_questions(iid, rng, a.absent_per_image)
        if iid in pool:
            qs += [q for k, q in enumerate(pool[iid]["questions"]) if (iid, k) in rep_keep]
        seen, uq = set(), []
        for q in qs:
            k = q["text"].lower()
            if k not in seen:
                seen.add(k); uq.append(q)
        if not uq:
            continue
        im = img_info[iid]
        base = dict(src="natural_mb", img=f"coco/train2014/{im['file_name']}", wh=[im["width"], im["height"]],
                    max_vis=640, plen=plen_image(im["width"], im["height"], 640), image_id=iid)
        rng.shuffle(uq)
        for i in range(0, len(uq), 48):
            recs.append(dict(base, questions=uq[i:i + 48]))
        for q in uq:
            cnt[(q.get("src", "?"), min(q.get("n_targets", 1 if q["present"] else 0), 4))] += 1
    rng.shuffle(recs)
    with open(f"{a.out}/train_mb.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    # calibration split: G + C on the v1.1 calibration images
    crecs = []
    for iid in sorted(calib):
        qs = G.get(iid, []) + cat_questions(iid, rng, a.absent_per_image)
        if not qs:
            continue
        im = img_info[iid]
        crecs.append(dict(src="calib_mb", img=f"coco/train2014/{im['file_name']}", wh=[im["width"], im["height"]],
                          max_vis=640, plen=plen_image(im["width"], im["height"], 640), image_id=iid, questions=qs))
    with open(f"{a.out}/calib_mb.jsonl", "w") as f:
        for r in crecs:
            f.write(json.dumps(r) + "\n")

    # eval: C on the gRefCOCO val images (eval format of dv/data/prep.py)
    val_imgs = sorted({json.loads(l)["image_id"] for l in open(f"{a.prep}/eval_grefcoco_val.jsonl")})
    erng = random.Random(1234)
    ecnt = collections.Counter()
    with open(f"{a.prep}/eval_cocoall_val.jsonl", "w") as f:
        for iid in val_imgs:
            qs = cat_questions(iid, erng, 1)
            if not qs:
                continue
            im = img_info[iid]
            for q in qs:
                ecnt[min(q["n_targets"], 4)] += 1
            f.write(json.dumps(dict(image_id=iid, file=im["file_name"], W=im["width"], H=im["height"],
                                    questions=[dict(text=q["text"], present=q["present"], box=q["box"], boxes=q["boxes"],
                                                    n_targets=q["n_targets"]) for q in qs])) + "\n")
    st = dict(train_records=len(recs), train_questions=sum(cnt.values()), grefcoco_multi_train_sents=n_g_all,
              by_src_and_n={f"{k[0]}|n={k[1]}": v for k, v in sorted(cnt.items())},
              calib_records=len(crecs), calib_questions=sum(len(r["questions"]) for r in crecs),
              eval_cocoall_val_by_n={str(k): v for k, v in sorted(ecnt.items())},
              held_out_images=len(held), calib_images=len(calib),
              leak_check=dict(train_eval_overlap=len({r["image_id"] for r in recs} & held),
                              train_calib_overlap=len({r["image_id"] for r in recs} & calib)))
    json.dump(st, open(f"{a.out}/stats.json", "w"), indent=1)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
