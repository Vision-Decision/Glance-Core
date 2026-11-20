"""Templated functional / generic description shift from the affordance lexicon (no LLM), TRAIN pool images only.

Per image (pool_natural.jsonl, i.e. RefCOCO/+/g + gRefCOCO train images with every eval image removed):
  ufunc      present: a non-person object whose category is unique in the image, described by an affordance phrase of
             a class that no OTHER object in the image satisfies (EN or ZH)          + literal anchor "the <cat>" (ulit)
  ugeneric   present: supercategory word, if the object's supercategory is unique among the image's objects
  absfunc    absent : an affordance phrase of a class NO object in the image satisfies
  absgeneric absent : supercategory word of a supercategory with no object in the image
  func       present: an original expression with the category name replaced by the class noun phrase
             ("left chair" -> "left seat"), grouped with the original expression (anchor)
--holdout_cats: categories never used as targets/classes in training (generalization test to unseen affordances).
Output records have the build_aug_pool.py format (orig questions are anchors; aug-tagged extra questions).
"""
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from affordance_lexicon import AFF, GENERIC, SKIP_GENERIC, SYN  # noqa: E402

MAIN = "~/Developer/decision-vison"
ap = argparse.ArgumentParser()
ap.add_argument("--pool", default=f"{MAIN}/data/prep_v1/pool_natural.jsonl")
ap.add_argument("--ann", default=f"{MAIN}/data/ann")
ap.add_argument("--exclude_images", default="", help="json list of image ids to skip (e.g. the LLM-augmented ones)")
ap.add_argument("--n_images", type=int, default=8000)
ap.add_argument("--holdout_cats", default="")
ap.add_argument("--zh_frac", type=float, default=0.5)
ap.add_argument("--funcattr_frac", type=float, default=0.3)
ap.add_argument("--p_absfunc", type=float, default=0.6)
ap.add_argument("--p_absgeneric", type=float, default=0.3)
ap.add_argument("--max_orig_groups", type=int, default=8, help="original description groups kept per image")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = random.Random(a.seed)
hold = {c.strip() for c in a.holdout_cats.split(",") if c.strip()}

coco = json.load(open(f"{a.ann}/annotations/instances_train2014.json"))
cats = {c["id"]: c for c in coco["categories"]}
img_info = {im["id"]: im for im in coco["images"]}
img_anns = collections.defaultdict(list)
for an in coco["annotations"]:
    img_anns[an["image_id"]].append(an)
recs = collections.defaultdict(list)
for l in open(a.pool):
    r = json.loads(l); recs[r["image_id"]].append(r)
excl = set(json.load(open(a.exclude_images))) if a.exclude_images else set()
imgs = sorted(set(recs) - excl); rng.shuffle(imgs); imgs = imgs[: a.n_images]
AFF_OK = {k: v for k, v in AFF.items() if not (set(v[0]) & hold)}
GEN_SUP = {k: v for k, v in GENERIC.items()}


def pick(en, zh):
    if rng.random() < a.zh_frac:
        return rng.choice(zh), "zh"
    return rng.choice(en), "en"


def boxkey(b):
    return "b" + ",".join(f"{v:.0f}" for v in b)


st, out = collections.Counter(), []
for iid in imgs:
    im = img_info[iid]; A = im["width"] * im["height"]
    anns = img_anns[iid]
    present = collections.Counter(cats[an["category_id"]]["name"] for an in anns)
    psup = collections.Counter(cats[an["category_id"]]["supercategory"] for an in anns)
    base = {k: v for k, v in recs[iid][0].items() if k != "questions"}
    groups = collections.OrderedDict()
    seen = set()
    for r in recs[iid]:
        for q in r["questions"]:
            if q["text"].lower() in seen:
                continue
            seen.add(q["text"].lower())
            k = boxkey(q["box"]) if q["present"] else "t" + q["text"].lower()
            if (k not in groups and len(groups) >= a.max_orig_groups) or len(groups.get(k, [])) >= 3:
                continue
            groups.setdefault(k, []).append(dict(q, grp=k, anchor=True, aug=None)); st["orig"] += 1
    # ---- present: unique-category objects
    cand = [an for an in anns if present[cats[an["category_id"]]["name"]] == 1 and not an["iscrowd"] and an["area"] / A >= 0.015
            and cats[an["category_id"]]["name"] not in hold | {"person"}]
    rng.shuffle(cand)
    for an in cand[:2]:
        c = cats[an["category_id"]]["name"]; sup = cats[an["category_id"]]["supercategory"]
        x, y, w, h = an["bbox"]; box = [x, y, x + w, y + h]; k = boxkey(box)
        others = present - collections.Counter({c: 1})
        cls = [n for n, v in AFF_OK.items() if c in v[0] and not any(o in v[0] for o in others)]
        g = groups.setdefault(k, [])
        added = False
        for n in rng.sample(cls, min(2, len(cls))):
            t, lang = pick(AFF_OK[n][1], AFF_OK[n][2])
            g.append(dict(type="box", text=t, present=True, box=box, src="lex", grp=k, anchor=False, aug="ufunc", lang=lang, cls=n))
            st["ufunc"] += 1; added = True
        if sup in GEN_SUP and psup[sup] == 1 and c not in SKIP_GENERIC:
            t, lang = pick(*GEN_SUP[sup])
            g.append(dict(type="box", text=t, present=True, box=box, src="lex", grp=k, anchor=False, aug="ugeneric", lang=lang))
            st["ugeneric"] += 1; added = True
        if not g:
            del groups[k]
        if added and not any(q["anchor"] for q in g):
            g.append(dict(type="box", text=f"the {c}", present=True, box=box, src="lex", grp=k, anchor=True, aug="ulit")); st["ulit"] += 1
    # ---- present: name -> affordance noun phrase inside original expressions
    for k, g in list(groups.items()):
        if not g:
            continue
        q = g[0]
        if not q["present"] or rng.random() > a.funcattr_frac:
            continue
        an = None
        for x in anns:  # category of the referred object
            bx = x["bbox"]
            if abs(bx[0] - q["box"][0]) < 2 and abs(bx[1] - q["box"][1]) < 2 and abs(bx[0] + bx[2] - q["box"][2]) < 2:
                an = x; break
        if an is None:
            continue
        c = cats[an["category_id"]]["name"]
        if c in hold:
            continue
        others = present - collections.Counter({c: 1})
        cls = [n for n, v in AFF_OK.items() if c in v[0] and v[3] and not any(o in v[0] for o in others if o != c)]
        if not cls:
            continue
        n = rng.choice(cls)
        for orig in [x for x in g if x["anchor"]][:1]:
            for s in sorted(SYN.get(c, [c]), key=len, reverse=True):
                m = re.search(r"\b" + re.escape(s) + r"s?\b", orig["text"], flags=re.I)
                if m:
                    t = orig["text"][: m.start()] + AFF_OK[n][3] + orig["text"][m.end():]
                    g.append(dict(type="box", text=t, present=True, box=orig["box"], src="lex", grp=k, anchor=False, aug="func", lang="en", cls=n))
                    st["func"] += 1
                    break
    # ---- absent
    abs_cls = [n for n, v in AFF_OK.items() if not any(o in v[0] for o in present)]
    for n in rng.sample(abs_cls, min(1 if rng.random() < a.p_absfunc else 0, len(abs_cls))):
        t, lang = pick(AFF_OK[n][1], AFF_OK[n][2]); k = "afunc_" + n
        groups.setdefault(k, []).append(dict(type="box", text=t, present=False, box=None, src="lex", neg="hard", grp=k, anchor=False,
                                             aug="absfunc", lang=lang, cls=n)); st["absfunc"] += 1
    abs_sup = [s for s in GEN_SUP if psup[s] == 0]
    if abs_sup and rng.random() < a.p_absgeneric:
        s = rng.choice(abs_sup); t, lang = pick(*GEN_SUP[s]); k = "agen_" + s
        groups.setdefault(k, []).append(dict(type="box", text=t, present=False, box=None, src="lex", neg="hard", grp=k, anchor=False,
                                             aug="absgeneric", lang=lang)); st["absgeneric"] += 1
    gl = [g for g in groups.values() if g]; rng.shuffle(gl)
    cur = []
    for g in gl:
        if cur and len(cur) + len(g) > 48:
            out.append(dict(base, src="natural_aug", questions=cur)); cur = []
        cur += g[:48]
    if cur:
        out.append(dict(base, src="natural_aug", questions=cur))
    st["images"] += 1
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
st["records"] = len(out)
print(json.dumps(st, indent=1))
