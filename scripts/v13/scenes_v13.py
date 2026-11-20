"""v1.3 negatives, step 1 (rules): tasks for the negative kinds v1.2 lacked (outputs/v1.2/regression_diagnosis.md).
The answer is always "absent"; catfree_gen.py writes the wording and its blind verifier must answer "none".

  rel_absent   class word + relation + named anchor, the class has NO annotated member ("the animal left of the bench")
  rel_thing    "thing / 东西" + relation + anchor, where no other annotated object satisfies even the relaxed relation and the
               side region is not a sliver of the image (the v1.2 data only ever had "thing + relation" as a positive)
  pos_absent   position word / superlative + a category that is absent (and none of its look-alikes present)
               ("the suitcase on the left side", "最左边的长颈鹿") -- the counterpart of pos_pos_sole
  generic_neg  class word of an absent class (more of v1.2's kind, to bring generic pos:neg to ~1.5:1)

  python scripts/v13/scenes_v13.py --ann ~/gv2/ann --gd ~/gv2/glance_data --exclude_ids probe_ids.json,robust_ids.json \
      --n 30000 --out scenes_v13.jsonl
Same decontamination as catfree_scenes.py (all eval images out); Glance-Data images allowed (they are training images).
"""
import argparse, collections, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v12"))
sys.path.insert(0, os.path.join(HERE, "..", "ground_v2"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from catfree_scenes import CLASSES, CLASSES_OF, REL_EN, describable, rel_holds  # noqa: E402
from lexicon import EN, SIMILAR  # noqa: E402

POS_WORDS = ["left", "right", "leftmost", "rightmost", "biggest", "smallest", "front", "top", "bottom"]


def side_room(r, B, W, H, frac=0.2):
    x1, y1, x2, y2 = B["box"]
    return {"left": x1 > frac * W, "right": x2 < (1 - frac) * W, "above": y1 > frac * H, "below": y2 < (1 - frac) * H,
            "on": True}[r]


def tasks_for(objs, W, H, rng, a):
    cnt = collections.Counter(o["cat"] for o in objs)
    present = set(cnt)
    T = []
    lang = lambda: "zh" if rng.random() < a.zh_frac else "en"
    members = collections.defaultdict(list)
    for o in objs:
        for k in CLASSES_OF.get(o["cat"], []):
            members[k].append(o)
    anchors = [o for o in objs if describable(o, cnt) and cnt[o["cat"]] == 1 and o["area"] >= 0.02]
    rng.shuffle(anchors)
    absent_cls = [k for k in CLASSES if k != "thing" and k not in members]
    # rel_absent
    for B in anchors[: a.max_rel_absent]:
        if not absent_cls:
            break
        k = rng.choice(absent_cls)
        r = rng.choice([x for x in REL_EN if x != "on" or B["cat"] in ("dining table", "bed", "couch", "bench", "chair")])
        T.append(dict(kind="rel_neg", anchor=B["id"], rel=r, cls=k, n_cls=0, sub="rel_absent", lang=lang()))
    # rel_thing
    cands = []
    for B in anchors:
        for r in ("left", "right", "above", "below"):
            if not side_room(r, B, W, H):
                continue
            if any(o["id"] != B["id"] and rel_holds(r, o["box"], B["box"], relaxed=True) for o in objs):
                continue
            cands.append((B, r))
    rng.shuffle(cands)
    for B, r in cands[: a.max_rel_thing]:
        T.append(dict(kind="rel_neg", anchor=B["id"], rel=r, cls="thing", n_cls=len(objs), sub="rel_thing", lang=lang()))
    # pos_absent
    absent = [c for c in EN if c not in present and c != "person" and not (set(SIMILAR.get(c, [])) & present)]
    for c in rng.sample(absent, min(a.max_pos_absent, len(absent))):
        T.append(dict(kind="pos_absent", absent_cat=c, position=rng.choice(POS_WORDS), lang=lang()))
    # generic_neg
    for k in rng.sample(absent_cls, min(a.max_generic_neg, len(absent_cls))):
        T.append(dict(kind="generic_neg", cls=k, lang=lang()))
    for i, t in enumerate(T):
        t["tid"] = f"t{i}"
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True); ap.add_argument("--gd", required=True)
    ap.add_argument("--exclude_ids", default=""); ap.add_argument("--skip_scenes", default="")
    ap.add_argument("--n", type=int, default=30000); ap.add_argument("--zh_frac", type=float, default=0.3)
    ap.add_argument("--max_rel_absent", type=int, default=1); ap.add_argument("--max_rel_thing", type=int, default=1)
    ap.add_argument("--max_pos_absent", type=int, default=1); ap.add_argument("--max_generic_neg", type=int, default=1)
    ap.add_argument("--seed", type=int, default=13); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    from build_scenes import held_out_ids
    coco = json.load(open(f"{a.ann}/annotations/instances_train2014.json"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns = collections.defaultdict(list)
    for an in coco["annotations"]:
        anns[an["image_id"]].append(an)
    imgs = {im["id"]: im for im in coco["images"]}
    held = held_out_ids(a.ann) | {im["id"] for im in json.load(open(f"{a.ann}/annotations/instances_val2017.json"))["images"]}
    for f in filter(None, a.exclude_ids.split(",")):
        held |= set(json.load(open(f)))
    for f in filter(None, a.skip_scenes.split(",")):
        held |= {json.loads(l)["image_id"] for l in open(f)}
    pool = sorted(i for i in imgs if i not in held and anns[i])
    rng.shuffle(pool)
    st = collections.Counter()
    n = 0
    with open(a.out, "w") as f:
        for iid in pool:
            if n >= a.n:
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
            if not T:
                continue
            for t in T:
                st[t.get("sub", t["kind"])] += 1
            f.write(json.dumps(dict(image_id=iid, file=im["file_name"], W=W, H=H, objs=objs, tasks=T)) + "\n")
            n += 1
    st["images"] = n
    print(json.dumps(dict(st), indent=1))


if __name__ == "__main__":
    main()
