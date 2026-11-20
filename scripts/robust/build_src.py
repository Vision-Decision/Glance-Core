"""Source items for the robustness eval set `Vision-Decision/glance-robust-ref` (text is generated later by gen_text.py).

Inputs (local, metadata only): data/ann/{refcoco,refcocoplus,refcocog}/validation.parquet, data/ann/grefcoco/grefs_unc.json,
data/ann/annotations/instances_train2014.json.
Output: one JSONL of source items (no generated text yet) with the LLM context each type needs.

Types (each N_POS present + N_ABS absent items):
  zh     Chinese translation of a RefCOCO/+/g val expression; absent = translated gRefCOCO-val no-target expression
  para   English paraphrase of the same; absent = paraphrased gRefCOCO no-target expression
  func   functional / affordance description ("something to drink from"); target category unique in the image, not person;
         absent = functional description of a category with no instance in the image (LLM checks it fits no present object)
  vague  generic noun ("the animal", "那个吃的") alone (hyper) or + relation to another unique object (rel), the target's
         COCO supercategory is unique in the image so the generic noun resolves; absent = generic noun of an absent supercategory
func and vague are ~30% Chinese (lang field). Every item keeps the original expression and the GT box (xyxy, pixels).
Run: uv run --with pandas --with pyarrow python scripts/robust/build_src.py --out <dir>/src.jsonl
"""
import argparse, collections, json, random
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ann", default="data/ann")
ap.add_argument("--out", required=True)
ap.add_argument("--n_pos", type=int, default=2000)
ap.add_argument("--n_abs", type=int, default=500)
ap.add_argument("--zh_frac", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)

coco = json.load(open(f"{a.ann}/annotations/instances_train2014.json"))
cats = {c["id"]: c for c in coco["categories"]}
img_info = {im["id"]: im for im in coco["images"]}
img_anns = collections.defaultdict(list)
for an in coco["annotations"]:
    img_anns[an["image_id"]].append(an)


def nbox(b, W, H):  # xyxy pixels -> 0..100 ints for the LLM
    return [round(100 * b[0] / W), round(100 * b[1] / H), round(100 * b[2] / W), round(100 * b[3] / H)]


def objects(iid):
    im = img_info[iid]; W, H = im["width"], im["height"]
    out = []
    for an in img_anns[iid]:
        x, y, w, h = an["bbox"]
        out.append(dict(ann_id=an["id"], cat=cats[an["category_id"]]["name"], sup=cats[an["category_id"]]["supercategory"],
                        box=[x, y, x + w, y + h], nbox=nbox([x, y, x + w, y + h], W, H), area=w * h / (W * H)))
    return out


# ---- positive pool: one random sentence per referred object, RefCOCO / RefCOCO+ / RefCOCOg val
pos = []
for ds in ["refcoco", "refcocoplus", "refcocog"]:
    df = pd.read_parquet(f"{a.ann}/{ds}/validation.parquet")
    for r in df.itertuples():
        info = json.loads(r.raw_image_info)
        s = rng.choice(list(r.sentences))
        pos.append(dict(src=ds, image_id=int(r.image_id), file=info["file_name"], W=info["width"], H=info["height"],
                        ann_id=int(r.ann_id), cat_id=int(r.category_id), box=[float(x) for x in r.bbox],
                        orig=" ".join(s["sent"].split()), sent_id=int(s["sent_id"])))
rng.shuffle(pos)
print("positive refs", len(pos), collections.Counter(p["src"] for p in pos))

# ---- absent pool: gRefCOCO val no-target expressions
grefs = json.load(open(f"{a.ann}/grefcoco/grefs_unc.json"))
neg = []
for r in grefs:
    if r["split"] != "val" or not r["no_target"]:
        continue
    im = img_info[r["image_id"]]
    s = rng.choice(r["sentences"])
    neg.append(dict(src="grefcoco", image_id=r["image_id"], file=im["file_name"], W=im["width"], H=im["height"],
                    orig=" ".join(s["sent"].split()), sent_id=int(s["sent_id"])))
rng.shuffle(neg)
print("no-target refs", len(neg))

items = []


def add(typ, base, present, k, **kw):
    it = dict(id=f"{typ}_{'p' if present else 'a'}{k:04d}", type=typ, present=present, image_id=base["image_id"],
              file=base["file"], W=base["W"], H=base["H"], src=base["src"], orig=base.get("orig"), sent_id=base.get("sent_id"),
              box=base.get("box") if present else None, cat=cats[base["cat_id"]]["name"] if present and "cat_id" in base else None)
    it.update(kw)
    items.append(it)


def strat(pool, n):  # balanced over the three source datasets
    by = collections.defaultdict(list)
    for p in pool:
        by[p["src"]].append(p)
    out, i = [], 0
    while len(out) < n:
        for k in sorted(by):
            if i < len(by[k]) and len(out) < n:
                out.append(by[k][i])
        i += 1
    return out


# zh / para: disjoint samples
zh_p = strat(pos[: len(pos) // 2], a.n_pos); para_p = strat(pos[len(pos) // 2:], a.n_pos)
for k, p in enumerate(zh_p):
    add("zh", p, True, k, lang="zh", sub="translate")
for k, p in enumerate(neg[: a.n_abs]):
    add("zh", p, False, k, lang="zh", sub="translate")
for k, p in enumerate(para_p):
    add("para", p, True, k, lang="en", sub="paraphrase")
for k, p in enumerate(neg[a.n_abs: 2 * a.n_abs]):
    add("para", p, False, k, lang="en", sub="paraphrase")


def ctx(iid, exclude=None):
    """other objects in the image (category + coarse 0-100 box), largest first, max 15"""
    ob = sorted([o for o in objects(iid) if o["ann_id"] != exclude and o["area"] > 0.002], key=lambda o: -o["area"])
    return [dict(cat=o["cat"], box=o["nbox"]) for o in ob[:15]]


def lang():
    return "zh" if rng.random() < a.zh_frac else "en"


# COCO objects in the val images (any annotated object, not only RefCOCO referents), for the "unique" sub-types.
# RefCOCO/+/g referents always share their category with another object in the image, so pure function / generic-noun
# queries are built on other COCO objects whose category (func) or supercategory (vague) is unique in the image.
def relation(t, an):
    tc = ((t[0] + t[2]) / 2, (t[1] + t[3]) / 2); ac = ((an[0] + an[2]) / 2, (an[1] + an[3]) / 2)
    dx, dy = tc[0] - ac[0], tc[1] - ac[1]
    if abs(dx) >= abs(dy):
        return "left of" if dx < 0 else "right of"
    return "above" if dy < 0 else "below"


val_ids = sorted({p["image_id"] for p in pos})
rng.shuffle(val_ids)
uniq_cat, uniq_sup = [], []
for iid in val_ids:
    ob = objects(iid)
    cc = collections.Counter(o["cat"] for o in ob); sc = collections.Counter(o["sup"] for o in ob)
    im = img_info[iid]
    base = lambda o: dict(src="coco", image_id=iid, file=im["file_name"], W=im["width"], H=im["height"], ann_id=o["ann_id"],
                          cat_id=[c for c in cats if cats[c]["name"] == o["cat"]][0], box=o["box"], orig=None, sent_id=None)
    big = [o for o in ob if o["area"] >= 0.015 and not any(an["iscrowd"] for an in img_anns[iid] if an["id"] == o["ann_id"])]
    uc = [o for o in big if cc[o["cat"]] == 1 and o["cat"] != "person"]
    if uc:
        o = rng.choice(uc)
        anchors = [x for x in ob if x["ann_id"] != o["ann_id"] and cc[x["cat"]] == 1 and x["area"] > 0.01]
        uniq_cat.append((base(o), o, ob, anchors))
    us = [o for o in big if sc[o["sup"]] == 1 and o["sup"] not in ("outdoor", "indoor", "accessory")]
    if us:
        o = rng.choice(us)
        uniq_sup.append((base(o), o, ob, None))
print("images with a unique-category object", len(uniq_cat), "unique-supercategory object", len(uniq_sup))
attr_pool = [p for p in pos[: len(pos) // 2]][::-1]  # RefCOCO referents for the *_attr sub-types

# func: 50% function-only on a unique-category object; 50% function phrase + the original attributes/location
nf_only = a.n_pos // 2
k = 0
for b, o, ob, _ in uniq_cat[:nf_only]:
    add("func", b, True, k, lang=lang(), sub="function_only", others=ctx(b["image_id"], o["ann_id"]), target_nbox=o["nbox"]); k += 1
for p in attr_pool:
    if k >= a.n_pos:
        break
    if cats[p["cat_id"]]["name"] == "person":
        continue
    add("func", p, True, k, lang=lang(), sub="function_attr", others=ctx(p["image_id"], p["ann_id"]),
        target_nbox=nbox(p["box"], p["W"], p["H"])); k += 1

# vague: 1/3 generic noun on a unique-supercategory object; 1/3 generic noun + relation to another unique object
# (target category unique too); 1/3 generic/loose noun ("that one", "the kid") + the original attributes/location
n3 = a.n_pos // 3
k = 0
for b, o, ob, _ in uniq_sup[:n3]:
    add("vague", b, True, k, lang=lang(), sub="generic", supercat=o["sup"], target_nbox=o["nbox"]); k += 1
for b, o, ob, anchors in uniq_cat[::-1]:  # reversed: relational targets mostly differ from function_only ones
    if k >= 2 * n3:
        break
    # anchor must not overlap the target (center-offset directions are meaningless for nested / overlapping boxes)
    anchors = [x for x in (anchors or []) if not (x["nbox"][0] < o["nbox"][2] and o["nbox"][0] < x["nbox"][2]
                                                  and x["nbox"][1] < o["nbox"][3] and o["nbox"][1] < x["nbox"][3])]
    if not anchors:
        continue
    an = max(anchors, key=lambda x: x["area"])
    add("vague", b, True, k, lang=lang(), sub="relational", supercat=o["sup"], target_nbox=o["nbox"], anchor=an["cat"],
        anchor_nbox=an["nbox"], relation=relation(o["nbox"], an["nbox"])); k += 1
for p in attr_pool[::-1]:
    if k >= a.n_pos:
        break
    add("vague", p, True, k, lang=lang(), sub="generic_attr", supercat=cats[p["cat_id"]]["supercategory"],
        target_nbox=nbox(p["box"], p["W"], p["H"]), others=ctx(p["image_id"], p["ann_id"])); k += 1


# absent func / vague: random val images, a category / supercategory with no instance in the image
val_imgs = sorted({p["image_id"]: p for p in pos}.items())
rng.shuffle(val_imgs)
all_cats = [c["name"] for c in coco["categories"] if c["name"] != "person"]
sups = sorted({c["supercategory"] for c in coco["categories"]} - {"outdoor", "indoor", "accessory"})
for typ in ("func", "vague"):
    k = 0
    for iid, p in val_imgs[(0 if typ == "func" else len(val_imgs) // 2):]:
        if k >= a.n_abs * 2:  # over-generate: the LLM may reject some (absent target must not fit any present object)
            break
        ob = objects(iid)
        present_c = {o["cat"] for o in ob}; present_s = {o["sup"] for o in ob}
        if typ == "func":
            cands = [c for c in all_cats if c not in present_c and cats[[i for i in cats if cats[i]["name"] == c][0]]["supercategory"] not in present_s]
            if not cands:
                continue
            add("func", p, False, k, orig=None, sent_id=None, lang=lang(), sub="function", absent_cat=rng.choice(cands), others=ctx(iid))
        else:
            cands = [s for s in sups if s not in present_s]
            if not cands:
                continue
            kw = dict(orig=None, sent_id=None, lang=lang(), absent_supercat=rng.choice(cands), others=ctx(iid))
            cc = collections.Counter(o["cat"] for o in ob)
            anchors = [o for o in ob if cc[o["cat"]] == 1 and o["area"] > 0.01]
            if k % 2 and anchors:
                an = max(anchors, key=lambda o: o["area"])
                kw.update(sub="relational", anchor=an["cat"], relation=rng.choice(["left of", "right of", "on", "next to", "behind"]))
            else:
                kw.update(sub="generic")
            add("vague", p, False, k, **kw)
        k += 1

with open(a.out, "w") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(collections.Counter((it["type"], it["present"]) for it in items))
