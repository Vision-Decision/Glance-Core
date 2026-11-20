"""Module 1 (rules): merge each image's annotations, choose and number the objects the LLM will describe.

Sources per image: COCO instances (+ captions), RefCOCO/+/g train expressions (by ann_id), gRefCOCO train sentences
(single-, multi- and no-target). Output: one JSON line per image ("scene").

  python scripts/ground_v2/build_scenes.py --ann ~/gv2/ann --split train --n 200 --out scenes_pilot.jsonl
  python scripts/ground_v2/build_scenes.py --ann ~/gv2/ann --split test  --n 300 --out scenes_test.jsonl \
      --exclude_ids probe_ids.json

Decontamination (train split): the pool is COCO train2014 minus every image of any eval split we report --
RefCOCO/+/g val/test*, gRefCOCO val/testA/testB (dv/data/prep.py's rule; covers Vision-Decision/glance-robust-ref, which is
built on RefCOCO/+/g val images), every COCO val2017 image (val2017 re-uses train2014 ids; the ambiguity / attribute
probes and our held-out test set live there), and --exclude_ids. The test split draws from val2017 minus --exclude_ids
(the probe images), so it was never seen by v1 / v1.1 (RefCOCO-train images only) nor by this pipeline's train split.

Object geometry flags (used by Module 3):
  marginal  : area < 0.2% of the image, or touches the border with area < 1% (mostly cut off / tiny) -> category-only,
              count and position questions about its category are skipped
  edge      : box touches the image border
"""
import argparse, collections, json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dv.data.prep import load_parquet_refs, norm_text  # noqa: E402

MAX_DESCRIBE = 6     # objects per image the LLM writes phrasings for
MAX_NUMBERED = 8     # objects drawn with a number (described ones + multi-target group members)


def held_out_ids(ann):
    split_files = {"val": "validation", "testA": "test", "testB": "testB", "test": "test"}
    held = set()
    for ds in ["refcoco", "refcocoplus", "refcocog"]:
        for sp in (["val", "testA", "testB"] if ds != "refcocog" else ["val", "test"]):
            held |= {q["image_id"] for q in load_parquet_refs(f"{ann}/{ds}/{split_files[sp]}.parquet", ds)}
    for r in json.load(open(f"{ann}/grefcoco/grefs_unc.json")):
        if r["split"] != "train":
            held.add(r["image_id"])
    return held


def geom(box, W, H):
    x1, y1, x2, y2 = box
    a = max(0.0, (x2 - x1) * (y2 - y1)) / (W * H)
    edge = x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2
    return a, edge, (a < 0.002) or (edge and a < 0.01)


def describable(o, objs, cnt):
    """an object the LLM may write a unique phrase for: not crowd / marginal, >= 1% of the image, not a fragment cut by
    the border (edge and < 3%), and its category is not a dense group (> 6 instances or any crowd box: "the tall brown
    vase" in a yard of vases cannot be made unique)"""
    if o["crowd"] or o["marginal"] or o["area"] < 0.01 or (o["edge"] and o["area"] < 0.03) or cnt[o["cat"]] > 6:
        return False
    return not any(x["crowd"] for x in objs if x["cat"] == o["cat"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--n", type=int, default=200, help="images (0 = whole pool)")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N images of the shuffled pool (shards)")
    ap.add_argument("--exclude_ids", default="", help="comma-separated JSON files with lists of image ids")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    excl = set()
    for f in filter(None, a.exclude_ids.split(",")):
        excl |= set(json.load(open(f)))

    year = "train2014" if a.split == "train" else "val2017"
    coco = json.load(open(f"{a.ann}/annotations/instances_{year}.json"))
    caps = collections.defaultdict(list)
    for c in json.load(open(f"{a.ann}/annotations/captions_{year}.json"))["annotations"]:
        caps[c["image_id"]].append(norm_text(c["caption"]))
    cats = {c["id"]: c for c in coco["categories"]}
    anns = collections.defaultdict(list)
    for an in coco["annotations"]:
        anns[an["image_id"]].append(an)
    imgs = {im["id"]: im for im in coco["images"]}

    refs = collections.defaultdict(list)       # ann_id -> human expressions
    g_multi = collections.defaultdict(list)    # image_id -> [(sent, [ann_id..])]
    g_none = collections.defaultdict(list)     # image_id -> [sent]
    if a.split == "train":
        val17 = {im["id"] for im in json.load(open(f"{a.ann}/annotations/instances_val2017.json"))["images"]}
        held = held_out_ids(a.ann) | val17 | excl
        import pandas as pd
        for ds in ["refcoco", "refcocoplus", "refcocog"]:
            df = pd.read_parquet(f"{a.ann}/{ds}/train.parquet")
            for r in df.itertuples():
                for s in r.sentences:
                    t = norm_text(s["sent"])
                    if t and t not in refs[int(r.ann_id)]:
                        refs[int(r.ann_id)].append(t)
        for r in json.load(open(f"{a.ann}/grefcoco/grefs_unc.json")):
            if r["split"] != "train":
                continue
            ids = [x for x in r["ann_id"] if x != -1] if isinstance(r["ann_id"], list) else [r["ann_id"]]
            for s in r["sentences"]:
                t = norm_text(s["sent"])
                if r["no_target"]:
                    g_none[r["image_id"]].append(t)
                elif len(ids) > 1:
                    g_multi[r["image_id"]].append((t, ids))
    else:
        held = excl
    pool = sorted(i for i in imgs if i not in held and anns[i])
    rng.shuffle(pool)
    print(f"{year}: {len(imgs)} images, held out {len(held & set(imgs))}, pool {len(pool)}", flush=True)
    pool = pool[a.offset:]
    out, n_img = open(a.out, "w"), 0
    for iid in pool:
        if a.n and n_img >= a.n:
            break
        im = imgs[iid]; W, H = im["width"], im["height"]
        objs = []
        for an in anns[iid]:
            x, y, w, h = an["bbox"]
            box = [x, y, x + w, y + h]
            area, edge, marg = geom(box, W, H)
            objs.append(dict(ann_id=an["id"], cat=cats[an["category_id"]]["name"], sup=cats[an["category_id"]]["supercategory"],
                             box=[round(v, 2) for v in box], area=round(area, 5), edge=edge, marginal=marg,
                             crowd=bool(an["iscrowd"]), refs=refs.get(an["id"], [])[:4]))
        cnt = collections.Counter(o["cat"] for o in objs)
        cand = [o for o in objs if describable(o, objs, cnt)]
        if not cand:
            continue
        # described objects: round-robin over categories (largest / human-referred first) for diversity
        by_cat = collections.defaultdict(list)
        for o in sorted(cand, key=lambda o: (-bool(o["refs"]), -o["area"])):
            by_cat[o["cat"]].append(o)
        order = sorted(by_cat, key=lambda c: -max(o["area"] for o in by_cat[c]))
        desc = []
        while len(desc) < MAX_DESCRIBE and any(by_cat.values()):
            for c in order:
                if by_cat[c] and len(desc) < MAX_DESCRIBE:
                    desc.append(by_cat[c].pop(0))
        for o in desc:
            o["describe"] = True
        # multi-target groups for the LLM: (a) two same-category instances that form a spatial pair (category n>=3,
        # the 2 left-most or 2 right-most with a clear gap to the third); (b) two described objects of different categories
        groups = []
        for c, n in cnt.items():
            inst = [o for o in objs if o["cat"] == c]
            if n < 3 or n > 8 or any(o["crowd"] or o["marginal"] for o in inst):
                continue
            inst.sort(key=lambda o: (o["box"][0] + o["box"][2]) / 2)
            cx = [(o["box"][0] + o["box"][2]) / 2 / W for o in inst]
            if cx[2] - cx[1] > 0.08 and cx[1] - cx[0] < cx[2] - cx[1]:
                groups.append(dict(kind="same_cat", members=[inst[0]["ann_id"], inst[1]["ann_id"]], side="left"))
            elif cx[-2] - cx[-3] > 0.08 and cx[-1] - cx[-2] < cx[-2] - cx[-3]:
                groups.append(dict(kind="same_cat", members=[inst[-2]["ann_id"], inst[-1]["ann_id"]], side="right"))
        dcats = [o for o in desc if cnt[o["cat"]] == 1]
        if len(dcats) >= 2:
            p = rng.sample(dcats, 2)
            groups.append(dict(kind="diff_cat", members=[p[0]["ann_id"], p[1]["ann_id"]]))
        groups = groups[:3]
        # number objects: described first, then group members not described
        num = [o for o in desc]
        for g in groups:
            for aid in g["members"]:
                o = next(x for x in objs if x["ann_id"] == aid)
                if o not in num:
                    num.append(o)
        if len(num) > MAX_NUMBERED:
            keep = {o["ann_id"] for o in num[:MAX_NUMBERED]}
            groups = [g for g in groups if all(m in keep for m in g["members"])]
            num = num[:MAX_NUMBERED]
        for k, o in enumerate(num):
            o["num"] = k + 1
        aid2num = {o["ann_id"]: o["num"] for o in num}
        for g in groups:
            g["nums"] = [aid2num[m] for m in g["members"]]
        sc = dict(image_id=iid, split=a.split, year=year, file=im["file_name"], W=W, H=H,
                  url=im.get("coco_url") or f"http://images.cocodataset.org/{year}/{im['file_name']}",
                  license=im.get("license"), captions=caps[iid][:5], objects=objs, groups=groups,
                  g_multi=[dict(text=t, ann_ids=ids) for t, ids in g_multi.get(iid, [])],
                  g_none=sorted(set(g_none.get(iid, []))))
        out.write(json.dumps(sc, ensure_ascii=False) + "\n")
        n_img += 1
    out.close()
    print("wrote", a.out, n_img, "scenes", flush=True)


if __name__ == "__main__":
    main()
