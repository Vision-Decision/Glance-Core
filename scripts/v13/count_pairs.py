"""v1.3 count minimal pairs (rules only): "two clocks" / "两个钟" as PRESENT questions on images that really have exactly that
many instances, so a count phrase is no longer a text-only cue for "absent" (arch-exp exp5: Glance-Data's count negatives are
bare "number + noun" strings a text-only probe rejects at P(present) = 0.014).

  python scripts/v13/count_pairs.py --ann data/ann --exclude_ids eval_image_ids.json --n_images 20000 --out count_pos.jsonl
Instances must be reliable: no crowd box of the category, every instance >= 0.5% of the image and not a border sliver,
no pile-prone category, 2 <= n <= 5. Questions carry every referent box (box = null: the single-box head trains presence
only; the multi-box head trains the set).
"""
import argparse, collections, json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ground_v2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lexicon import EN, EN_PL, ZH, ZH_MW, NUM, ZH_NUM  # noqa: E402
from dv.data.prep_v1 import plen_image  # noqa: E402

PILES = {"book", "banana", "carrot", "broccoli", "apple", "orange", "donut"}
EN_T = ["{num} {pl}", "the {num} {pl}", "all {num} {pl}", "{num} {pl} in the picture"]
ZH_T = ["{zn}{mw}{zh}", "那{zn}{mw}{zh}", "这{zn}{mw}{zh}", "图里的{zn}{mw}{zh}"]

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True); ap.add_argument("--exclude_ids", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n_images", type=int, default=20000); ap.add_argument("--zh_frac", type=float, default=0.4)
ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()
rng = random.Random(a.seed)
coco = json.load(open(f"{a.ann}/annotations/instances_train2014.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
imgs = {im["id"]: im for im in coco["images"]}
by = collections.defaultdict(lambda: collections.defaultdict(list))
for an in coco["annotations"]:
    by[an["image_id"]][cats[an["category_id"]]].append(an)
ev = json.load(open(a.exclude_ids))
held = set().union(*[set(v) for v in ev.values()])
pool = [i for i in imgs if i not in held and i in by]
rng.shuffle(pool)
st, n_img = collections.Counter(), 0
with open(a.out, "w") as f:
    for iid in pool:
        if n_img >= a.n_images:
            break
        im = imgs[iid]; W, H = im["width"], im["height"]
        qs = []
        for c, lst in by[iid].items():
            n = len(lst)
            if c in PILES or not 2 <= n <= 5 or any(x["iscrowd"] for x in lst):
                continue
            ok = True
            for x in lst:
                bx, by_, bw, bh = x["bbox"]
                area = bw * bh / (W * H)
                edge = bx <= 2 or by_ <= 2 or bx + bw >= W - 2 or by_ + bh >= H - 2
                if area < 0.005 or (edge and area < 0.02):
                    ok = False
            if not ok:
                continue
            boxes = [[round(x["bbox"][0], 1), round(x["bbox"][1], 1), round(x["bbox"][0] + x["bbox"][2], 1),
                      round(x["bbox"][1] + x["bbox"][3], 1)] for x in lst]
            if rng.random() < a.zh_frac:
                t, lang = rng.choice(ZH_T).format(zn=ZH_NUM[n], mw=ZH_MW[c], zh=ZH[c]), "zh"
            else:
                t, lang = rng.choice(EN_T).format(num=NUM[n], pl=EN_PL[c]), "en"
            qs.append(dict(type="box", text=t, present=True, box=None, boxes=boxes, n_targets=n, qtype="count_pos", lang=lang,
                           neg=None, src="count_rule"))
            st[f"n={n}"] += 1; st[lang] += 1
        if qs:
            f.write(json.dumps(dict(src="count_rule", img=f"coco/train2014/{im['file_name']}", wh=[W, H], max_vis=640,
                                    plen=plen_image(W, H, 640), image_id=iid, questions=qs), ensure_ascii=False) + "\n")
            n_img += 1
st["images"] = n_img
print(json.dumps(dict(st)))
