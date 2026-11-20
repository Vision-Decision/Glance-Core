"""Augmented natural training pool (TRAIN images only) from gen_aug.py caches.

For every selected image with finished generations, one or more records (<= 48 questions, whole description groups
kept together) holding the original v1 box questions plus tagged augmented ones:
  aug=zh | para        translation / paraphrase of every original question (present AND absent; same label / box)
  aug=func | vague     affordance rewrite / casual vague rewrite of ~40% of present questions (EN or ZH)
  aug=ufunc | ugeneric unique-category object: function-only / supercategory-word queries (EN + ZH), with a literal
                       anchor "the <category>" (aug=ulit) so the consistency loss has an anchor
  aug=absfunc | absgeneric  absent category / supercategory: function / generic queries (EN + ZH), present=false
Groups: present questions are grouped by target box (all original expressions of the object are anchors); absent
questions by their original text. Mixers select variants by the `aug` tag (orig questions have aug=None).
Usage: python build_aug_pool.py --aug <gen_aug out dir> --out aug_natural.jsonl
"""
import argparse, collections, json, random, re

MAIN = "~/Developer/decision-vison"
ap = argparse.ArgumentParser()
ap.add_argument("--pool", default=f"{MAIN}/data/prep_v1/pool_natural.jsonl")
ap.add_argument("--aug", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_q", type=int, default=48)
a = ap.parse_args()
rng = random.Random(0)
items = json.load(open(f"{a.aug}/items.json"))
cache = {}
for t in ("expr", "unique"):
    try:
        for l in open(f"{a.aug}/cache_{t}.jsonl"):
            d = json.loads(l); cache[d["id"]] = d
    except FileNotFoundError:
        pass
zhre = re.compile(r"[一-鿿]")
by_img_items = collections.defaultdict(list)
for it in items["expr"]:
    by_img_items[it["image_id"]].append(it)
uniq = {u["image_id"]: u for u in items["unique"]}
recs = collections.defaultdict(list)
for l in open(a.pool):
    r = json.loads(l)
    recs[r["image_id"]].append(r)

out, st = [], collections.Counter()
for iid, its in by_img_items.items():
    if not all(it["id"] in cache for it in its):  # only images whose expressions are all generated
        st["img_incomplete"] += 1; continue
    base = {k: v for k, v in recs[iid][0].items() if k != "questions"}
    gen = {it["text"].lower(): cache[it["id"]] for it in its}
    groups = collections.OrderedDict()

    def gkey(q):
        return "b" + ",".join(f"{v:.0f}" for v in q["box"]) if q["present"] else "t" + q["text"].lower()

    seen = set()
    for r in recs[iid]:
        for q in r["questions"]:
            if q["text"].lower() in seen:
                continue
            seen.add(q["text"].lower())
            k = gkey(q)
            g = groups.setdefault(k, [])
            g.append(dict(q, grp=k, anchor=True, aug=None)); st["orig"] += 1
            c = gen.get(q["text"].lower())
            if not c:
                continue
            for v in ("zh", "para", "func", "vague"):
                t = str(c.get(v) or "").strip()
                if not t or t.lower() == q["text"].lower() or (v in ("func", "vague") and not q["present"]):
                    continue
                if v == "zh" and not zhre.search(t):
                    continue
                g.append(dict(type="box", text=t, present=q["present"], box=q.get("box"), src=q.get("src"), neg=q.get("neg"),
                              grp=k, anchor=False, aug=v, lang="zh" if zhre.search(t) else "en"))
                st[v] += 1
    u = uniq.get(iid); c = cache.get(u["id"]) if u else None
    if c:
        if u.get("cat"):
            k = "b" + ",".join(f"{v:.0f}" for v in u["box"])
            g = groups.setdefault(k, [])
            if not any(q["anchor"] for q in g):
                g.append(dict(type="box", text=f"the {u['cat']}", present=True, box=u["box"], src="coco_unique", grp=k,
                              anchor=True, aug="ulit")); st["ulit"] += 1
            for v, f in (("ufunc", "func_en"), ("ufunc", "func_zh"), ("ugeneric", "generic_en"), ("ugeneric", "generic_zh")):
                t = str(c.get(f) or "").strip()
                if t and not re.search(r"\b" + re.escape(u["cat"]) + r"s?\b", t.lower()):
                    g.append(dict(type="box", text=t, present=True, box=u["box"], src="coco_unique", grp=k, anchor=False, aug=v,
                                  lang=f[-2:])); st[v] += 1
        for v, f, cn in (("absfunc", "absent_func_en", "absent_cat"), ("absfunc", "absent_func_zh", "absent_cat"),
                         ("absgeneric", "absent_generic_en", "absent_supercat"), ("absgeneric", "absent_generic_zh", "absent_supercat")):
            t = str(c.get(f) or "").strip()
            if t and u.get(cn) and not re.search(r"\b" + re.escape(u[cn]) + r"s?\b", t.lower()):
                k = "a" + v + f[-2:]
                groups.setdefault(k, []).append(dict(type="box", text=t, present=False, box=None, src="coco_absent", neg="hard",
                                                     grp=k, anchor=False, aug=v, lang=f[-2:])); st[v] += 1
    gl = list(groups.values()); rng.shuffle(gl)
    cur = []
    for g in gl:
        g = g[: a.max_q]
        if cur and len(cur) + len(g) > a.max_q:
            out.append(dict(base, src="natural_aug", questions=cur)); cur = []
        cur = cur + g
    if cur:
        out.append(dict(base, src="natural_aug", questions=cur))
    st["images"] += 1
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
st["records"] = len(out)
print(json.dumps(st, indent=1))
