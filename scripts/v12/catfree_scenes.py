"""v1.2 category-free grounding data, step 1 (rules): pick images and decide every answer before any LLM is called.

For each COCO train2014 image (eval-decontaminated, not used by Glance-Data) we list the annotated objects and emit
*tasks*. A task fixes the answer (target object, or "absent") and the kind of wording; the LLM (catfree_gen.py) only
writes the words, and a blind second pass re-identifies the referent before anything is kept.

Kinds (positives answer one annotated object; negatives answer "absent"):
  func_pos     pure function / affordance of the target ("something to drink from", "能坐的东西"); target is not a
               person / animal, the only instance of its category, describable
  generic_pos  generic class word only ("the animal", "那个家具"); the target is the ONLY annotated member of its class
  rel_pos      class word + spatial relation to a named, unique anchor ("the animal left of the fridge", "冰箱左边那只动物");
               the target satisfies the relation and no other member of its class satisfies even a relaxed version
  func_neg     function of a category that is absent (and none of its look-alikes present); verified absent on the image
  generic_neg  class word of a class with no annotated member ("the vehicle" in a kitchen)
  rel_neg      class word + relation where the class IS present but no member satisfies even a relaxed relation
               (the hard counterpart of rel_pos: the model must check the geometry, not the noun)

  python scripts/v12/catfree_scenes.py --ann ~/gv2/ann --gd ~/gv2/glance_data --exclude_ids ~/gv2/probe_ids.json,robust_ids.json \
      --n 24000 --out ~/v12/scenes.jsonl
"""
import argparse, collections, glob, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ground_v2"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lexicon import EN, ZH, ZH_MW, SIMILAR  # noqa: E402

# generic classes: (members, english words, chinese words). Words are suggestions for the writer; correctness comes from
# the membership rule + blind verification. Classes whose only natural words are category synonyms (bag, plant) are omitted.
CLASSES = {
    "animal": (["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
               ["animal", "creature"], ["动物"]),
    "vehicle": (["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
                ["vehicle", "means of transport"], ["交通工具"]),
    "furniture": (["chair", "couch", "bed", "dining table", "bench"], ["furniture", "piece of furniture"], ["家具"]),
    "food": (["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
             ["food", "something to eat"], ["食物", "吃的"]),
    "device": (["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"], ["device", "gadget", "electronic device"],
               ["电子设备", "电子产品"]),
    "appliance": (["microwave", "oven", "toaster", "refrigerator"], ["appliance", "kitchen appliance"], ["电器", "家电"]),
    "utensil": (["fork", "knife", "spoon"], ["utensil", "piece of cutlery"], ["餐具"]),
    "sports_gear": (["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                     "surfboard", "tennis racket"], ["sports equipment", "piece of sports gear"], ["运动器材", "体育用品"]),
    "kitchenware": (["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"], ["kitchen item", "kitchenware", "tableware"],
                    ["厨房用品", "餐具"]),
    "accessory": (["backpack", "umbrella", "handbag", "tie", "suitcase"], ["accessory", "personal item", "belonging"],
                  ["随身物品", "配饰"]),
    # every non-person object: only usable with a relation (alone it is never unique enough to be natural)
    "thing": ([c for c in EN if c != "person"], ["thing", "object", "item"], ["东西", "物品"]),
}
CLASSES_OF = {}
for _k, (_m, _, _) in CLASSES.items():
    for _c in _m:
        CLASSES_OF.setdefault(_c, []).append(_k)
NO_FUNC = {"person"}
SURFACE = {"dining table", "bed", "couch", "bench", "chair", "toilet", "suitcase", "sink", "oven", "refrigerator"}
REL_EN = {"left": "to the left of", "right": "to the right of", "above": "above", "below": "below", "on": "on"}
REL_OVERLAP = 0.3  # min overlap (fraction of the smaller box) on the other axis for left/right/above/below
REL_ZH = {"left": "左边", "right": "右边", "above": "上方", "below": "下方", "on": "上"}


def describable(o, cnt):
    return not (o["crowd"] or o["area"] < 0.01 or (o["edge"] and o["area"] < 0.03) or cnt[o["cat"]] > 6)


def _ov(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))


def rel_holds(r, A, B, relaxed=False):
    """spatial relation of box A to anchor box B (xyxy px). relaxed=True is the looser test applied to the other class
    members: anything that *might* be described by the relation disqualifies a positive / a negative."""
    ax1, ay1, ax2, ay2 = A; bx1, by1, bx2, by2 = B
    aw, ah, bw, bh = ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    if r in ("left", "right"):
        vo = _ov(ay1, ay2, by1, by2)
        if relaxed:
            return (acx < bx1 + 0.25 * bw) if r == "left" else (acx > bx2 - 0.25 * bw)
        side = (ax2 <= bx1 + 0.1 * bw and acx < bx1) if r == "left" else (ax1 >= bx2 - 0.1 * bw and acx > bx2)
        return side and vo > REL_OVERLAP * min(ah, bh)
    if r in ("above", "below"):
        ho = _ov(ax1, ax2, bx1, bx2)
        if relaxed:
            return (acy < by1 + 0.25 * bh) if r == "above" else (acy > by2 - 0.25 * bh)
        side = (ay2 <= by1 + 0.1 * bh and acy < by1) if r == "above" else (ay1 >= by2 - 0.1 * bh and acy > by2)
        return side and ho > REL_OVERLAP * min(aw, bw)
    if r == "on":
        inside_x = bx1 <= acx <= bx2
        if relaxed:
            return _ov(ax1, ax2, bx1, bx2) > 0.3 * aw and _ov(ay1, ay2, by1, by2) > 0.2 * ah
        return inside_x and by1 - 0.1 * bh <= ay2 <= by1 + 0.6 * bh and acy < (by1 + by2) / 2 and aw * ah < 0.5 * bw * bh
    raise ValueError(r)


def tasks_for(objs, W, H, rng, a):
    cnt = collections.Counter(o["cat"] for o in objs)
    present = set(cnt)
    T = []
    lang = lambda: "zh" if rng.random() < a.zh_frac else "en"
    good = [o for o in objs if describable(o, cnt)]
    # ---- func_pos: sole instance of a non-living category
    fp = [o for o in good if o["cat"] not in NO_FUNC and cnt[o["cat"]] == 1]
    rng.shuffle(fp)
    for o in fp[: a.max_func]:
        T.append(dict(kind="func_pos", target=o["id"], lang=lang()))
    # ---- generic_pos: only annotated member of its class
    members = collections.defaultdict(list)
    for o in objs:
        for k in CLASSES_OF.get(o["cat"], []):
            members[k].append(o)
    gp = [(k, l[0]) for k, l in members.items() if k != "thing" and len(l) == 1 and describable(l[0], cnt)]
    rng.shuffle(gp)
    used = set()
    for k, o in gp:
        if o["id"] in used or len(used) >= a.max_generic:
            continue
        used.add(o["id"])
        T.append(dict(kind="generic_pos", target=o["id"], cls=k, lang=lang()))
    # ---- relations: anchors are describable objects whose category is unique in the image (naming them is unambiguous)
    anchors = [o for o in good if cnt[o["cat"]] == 1 and o["area"] >= a.anchor_min_area]
    rp, rn = [], []
    for k, l in members.items():
        for B in anchors:
            if k != "thing" and B["cat"] in CLASSES[k][0]:  # no "the animal left of the dog"
                continue
            for r in REL_EN:
                if r == "on" and B["cat"] not in SURFACE:
                    continue
                strict = [o for o in l if o["id"] != B["id"] and rel_holds(r, o["box"], B["box"])]
                loose = [o for o in l if o["id"] != B["id"] and rel_holds(r, o["box"], B["box"], relaxed=True)]
                if len(strict) == 1 and len(loose) == 1 and describable(strict[0], cnt):
                    rp.append(dict(kind="rel_pos", target=strict[0]["id"], anchor=B["id"], rel=r, cls=k, n_cls=len(l)))
                elif not loose and l:
                    rn.append(dict(kind="rel_neg", anchor=B["id"], rel=r, cls=k, n_cls=len(l)))
    # prefer relations that actually disambiguate (class has >= 2 members); one task per target
    rng.shuffle(rp)
    rp.sort(key=lambda t: -min(t["n_cls"], 2))
    seen = set()
    for t in rp:
        if t["target"] in seen or len([x for x in T if x["kind"] == "rel_pos"]) >= a.max_rel:
            continue
        seen.add(t["target"]); t["lang"] = lang(); T.append(t)
    rng.shuffle(rn)
    for t in rn[: a.max_rel_neg]:
        t["lang"] = lang(); T.append(t)
    # ---- func_neg: an absent non-living category none of whose look-alikes is present
    absent = [c for c in EN if c not in present and c not in NO_FUNC and not (set(SIMILAR.get(c, [])) & present)]
    for c in rng.sample(absent, min(a.max_func_neg, len(absent))):
        T.append(dict(kind="func_neg", absent_cat=c, lang=lang()))
    # ---- generic_neg: a class with no annotated member
    empty = [k for k in CLASSES if k not in members and k != "thing"]
    for k in rng.sample(empty, min(a.max_generic_neg, len(empty))):
        T.append(dict(kind="generic_neg", cls=k, lang=lang()))
    for i, t in enumerate(T):
        t["tid"] = f"t{i}"
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--gd", required=True, help="Glance-Data dir (its images are excluded so this data adds new images)")
    ap.add_argument("--exclude_ids", default="")
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--zh_frac", type=float, default=0.3)
    ap.add_argument("--max_func", type=int, default=2)
    ap.add_argument("--max_generic", type=int, default=2)
    ap.add_argument("--max_rel", type=int, default=4)
    ap.add_argument("--max_rel_neg", type=int, default=2)
    ap.add_argument("--max_func_neg", type=int, default=1)
    ap.add_argument("--max_generic_neg", type=int, default=1)
    ap.add_argument("--rel_extra", type=int, default=25000)
    ap.add_argument("--rel_overlap", type=float, default=0.3)
    ap.add_argument("--anchor_min_area", type=float, default=0.02)
    ap.add_argument("--allow_gd", action="store_true", help="also use Glance-Data training images (relation pass)")
    ap.add_argument("--skip_scenes", default="", help="scenes.jsonl whose images are skipped (second, relation-only pass)")
    ap.add_argument("--min_pos_tasks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    import pandas as pd
    from build_scenes import held_out_ids
    coco = json.load(open(f"{a.ann}/annotations/instances_train2014.json"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns = collections.defaultdict(list)
    for an in coco["annotations"]:
        anns[an["image_id"]].append(an)
    imgs = {im["id"]: im for im in coco["images"]}
    excl = {"refcoco_grefcoco_eval": held_out_ids(a.ann),
            "val2017": {im["id"] for im in json.load(open(f"{a.ann}/annotations/instances_val2017.json"))["images"]}}
    gd = set()
    for f in glob.glob(f"{a.gd}/data/*.parquet"):
        gd |= set(int(x) for x in pd.read_parquet(f, columns=["image_id"]).image_id)
    if not a.allow_gd:
        excl["glance_data"] = gd
    for f in filter(None, a.exclude_ids.split(",")):
        excl[os.path.basename(f)] = set(json.load(open(f)))
    global REL_OVERLAP
    REL_OVERLAP = a.rel_overlap
    held = set().union(*excl.values())
    if a.skip_scenes:
        held |= {json.loads(l)["image_id"] for l in open(a.skip_scenes)}
    pool = sorted(i for i in imgs if i not in held and anns[i])
    rng.shuffle(pool)
    print({k: len(v & set(imgs)) for k, v in excl.items()}, "pool", len(pool), flush=True)
    st = collections.Counter()
    with open(a.out, "w") as f:
        n = n_extra = 0
        for iid in pool:
            extra = n >= a.n  # past --n: images only for relation tasks (the scarcest kind), up to --rel_extra of them
            if extra and n_extra >= a.rel_extra:
                break
            im = imgs[iid]; W, H = im["width"], im["height"]
            objs = []
            for an in anns[iid]:
                x, y, w, h = an["bbox"]
                if w < 1 or h < 1:
                    continue
                area = w * h / (W * H)
                edge = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
                objs.append(dict(id=len(objs) + 1, ann_id=an["id"], cat=cats[an["category_id"]], crowd=bool(an["iscrowd"]),
                                 box=[round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)], area=round(area, 4), edge=edge))
            if not objs or len(objs) > 40:
                continue
            T = tasks_for(objs, W, H, rng, a)
            if extra:
                T = [t for t in T if t["kind"] == "rel_pos"] + [t for t in T if t["kind"] == "rel_neg"][:1]
                if not any(t["kind"] == "rel_pos" for t in T):
                    continue
                n_extra += 1
            elif sum(t["kind"].endswith("_pos") for t in T) < a.min_pos_tasks:
                continue
            else:
                n += 1
            for t in T:
                st[t["kind"]] += 1
                st[f"{t['kind']}/{t['lang']}"] += 1
            f.write(json.dumps(dict(image_id=iid, file=im["file_name"], W=W, H=H, objs=objs, tasks=T, rel_only=extra)) + "\n")
    st["images"] = n; st["images_rel_only"] = n_extra
    print(json.dumps(dict(sorted(st.items())), indent=1))
    json.dump(dict(excluded={k: len(v) for k, v in excl.items()}, pool=len(pool), tasks=dict(st)),
              open(a.out.replace(".jsonl", "_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
