"""Paired description-shift eval set ("robust-groups"), built from VAL splits only.

Every group = one target (or one absent query) x several descriptions of it, so per-target consistency
("all descriptions correct") can be measured:
  present groups (RefCOCO / RefCOCO+ / RefCOCOg val referents):
      orig (the human expression), zh (Chinese translation), para (English paraphrase),
      func (object name replaced by its function / affordance, attributes + location kept; EN or ZH),
      vague (casual vague reference: 'the guy on the left', '左边那个'), zh_vague
  unique groups (COCO objects of val images whose category is unique in the image, not person):
      func_only (function only, no attributes: 'something to drink from' / '能喝的东西'), generic (supercategory word)
  absent groups (gRefCOCO val no-target expressions): orig, zh, para  (all absent)
  absent_func (val images; a COCO category + its supercategory with no instance): func_only EN/ZH (absent)
Output: {out}/groups.jsonl (one line per description) + eval prep files {out}/eval_rg_{kind}.jsonl (eval.py v1 format).
Run (local, annotations only): python scripts/sprint_robust/build_groups.py --out <dir> [--n 600]
"""
import argparse, collections, json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import chat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ann", default="~/Developer/decision-vison/data/ann")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=600, help="present groups (balanced over RefCOCO/+/g)")
ap.add_argument("--n_unique", type=int, default=300)
ap.add_argument("--n_abs", type=int, default=300)
ap.add_argument("--n_abs_func", type=int, default=300)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--workers", type=int, default=12)
a = ap.parse_args()
rng = random.Random(a.seed)
os.makedirs(a.out, exist_ok=True)

coco = json.load(open(f"{a.ann}/annotations/instances_train2014.json"))
cats = {c["id"]: c for c in coco["categories"]}
img_info = {im["id"]: im for im in coco["images"]}
img_anns = collections.defaultdict(list)
for an in coco["annotations"]:
    img_anns[an["image_id"]].append(an)


def others(iid, excl=None):
    ob = sorted([an for an in img_anns[iid] if an["id"] != excl and an["area"] / (img_info[iid]["width"] * img_info[iid]["height"]) > 0.002],
                key=lambda an: -an["area"])
    return sorted({cats[an["category_id"]]["name"] for an in ob[:15]})


# ---------------- present groups (val referents)
pos = collections.defaultdict(list)
for ds in ["refcoco", "refcocoplus", "refcocog"]:
    df = pd.read_parquet(f"{a.ann}/{ds}/validation.parquet")
    for r in df.itertuples():
        info = json.loads(r.raw_image_info)
        s = rng.choice(list(r.sentences))
        pos[ds].append(dict(src=ds, image_id=int(r.image_id), file=info["file_name"], W=info["width"], H=info["height"],
                            ann_id=int(r.ann_id), cat=cats[int(r.category_id)]["name"], box=[float(x) for x in r.bbox],
                            orig=" ".join(s["sent"].split())))
groups = []
for ds in pos:
    rng.shuffle(pos[ds])
    for p in pos[ds][: a.n // 3]:
        p["gid"] = f"P{len(groups):04d}"; p["kind"] = "present"; p["others"] = others(p["image_id"], p["ann_id"])
        groups.append(p)
val_imgs = sorted({p["image_id"] for ds in pos for p in pos[ds]})
rng.shuffle(val_imgs)

# ---------------- unique-category groups (val images)
n_u = 0
for iid in val_imgs:
    if n_u >= a.n_unique:
        break
    im = img_info[iid]; A = im["width"] * im["height"]
    cc = collections.Counter(an["category_id"] for an in img_anns[iid])
    cand = [an for an in img_anns[iid] if cc[an["category_id"]] == 1 and not an["iscrowd"] and an["area"] / A >= 0.015
            and cats[an["category_id"]]["name"] != "person"]
    if not cand:
        continue
    an = rng.choice(cand)
    x, y, w, h = an["bbox"]
    groups.append(dict(gid=f"U{n_u:04d}", kind="unique", src="coco", image_id=iid, file=im["file_name"], W=im["width"], H=im["height"],
                       ann_id=an["id"], cat=cats[an["category_id"]]["name"], supercat=cats[an["category_id"]]["supercategory"],
                       box=[x, y, x + w, y + h], orig=None, others=others(iid, an["id"])))
    n_u += 1

# ---------------- absent groups (gRefCOCO val no-target)
grefs = json.load(open(f"{a.ann}/grefcoco/grefs_unc.json"))
neg = [r for r in grefs if r["split"] == "val" and r["no_target"]]
rng.shuffle(neg)
for k, r in enumerate(neg[: a.n_abs]):
    im = img_info[r["image_id"]]
    groups.append(dict(gid=f"A{k:04d}", kind="absent", src="grefcoco", image_id=r["image_id"], file=im["file_name"], W=im["width"],
                       H=im["height"], box=None, orig=" ".join(rng.choice(r["sentences"])["sent"].split()), others=others(r["image_id"])))
# ---------------- absent functional (val images, category + supercategory absent)
all_c = [c for c in cats.values() if c["name"] != "person"]
for k, iid in enumerate(val_imgs[::-1][: a.n_abs_func]):
    im = img_info[iid]
    pc = {an["category_id"] for an in img_anns[iid]}; ps = {cats[c]["supercategory"] for c in pc}
    cand = [c["name"] for c in all_c if c["id"] not in pc and c["supercategory"] not in ps]
    if not cand:
        continue
    groups.append(dict(gid=f"F{k:04d}", kind="absent_func", src="coco", image_id=iid, file=im["file_name"], W=im["width"], H=im["height"],
                       box=None, orig=None, absent_cat=rng.choice(cand), others=others(iid)))
print(collections.Counter(g["kind"] for g in groups))

SYS = ("You write test queries for a visual-grounding model that must box one object in a COCO photo, or say it is absent. "
       "Return ONLY JSON {\"results\": [{\"gid\": ..., <fields>}]} with one entry per input gid. ")
TASK = {
    "present": (
        "For each item, `orig` is a human referring expression for an object of category `cat`; `others` lists other object "
        "categories in the photo. Write:\n"
        " zh: a natural colloquial Simplified Chinese translation of `orig` (keep every attribute/location cue, add nothing; "
        "everyday words: guy->那个男的, kid->小孩).\n"
        " para: an English paraphrase with the same meaning but different words (synonym for the head noun if one exists, "
        "different syntax); keep all cues.\n"
        " func: replace the object's name by what it is used for / what you can do with it (function or affordance), keep the "
        "attribute and location cues: 'red chair on left' -> 'the red thing to sit on, on the left'. Never use the category name, "
        "a synonym or a hypernym. If `cat` is person, describe the person by role/activity instead of 'man/woman/person' "
        "('the one holding the racket on the right'). Write it in English for odd-numbered gids and in Chinese for even ones "
        "(by the last digit).\n"
        " vague: how a casual user would say it: shorter/looser, a vague noun ('that one', 'the thing', 'the guy', 'the kid' only if "
        "implied by orig) plus just enough of the cues to still pick out the same object. English.\n"
        " zh_vague: the same kind of casual vague reference in Chinese ('左边那个', '中间那个小孩').\n"
        "If `orig` is unintelligible, set ok=false. Fields: gid, zh, para, func, vague, zh_vague, ok."),
    "unique": (
        "For each item the TARGET has category `cat` (supercategory `supercat`) and is the only object of its category in the "
        "photo; `others` lists the other categories present. Write:\n"
        " func_en: refer to the target ONLY by its function / affordance in English, 2-8 words ('something to drink from', "
        "'what I can sit on'). Never use the category name, a synonym or a close hypernym, and the query must not also fit any "
        "of `others` (if a bottle and a cup are both present, not 'something to drink'; use a more specific function).\n"
        " func_zh: the same in natural Chinese ('能喝水的东西', '可以坐的地方').\n"
        " generic_en: a vague generic noun at supercategory level or broader ('the animal', 'the vehicle', 'the food', 'that "
        "gadget'), 1-4 words, must not also fit any of `others`.\n"
        " generic_zh: the same in Chinese ('那只动物', '吃的').\n"
        "Set ok=false if no such query can avoid fitting `others`. Fields: gid, func_en, func_zh, generic_en, generic_zh, ok."),
    "absent": (
        "Each `orig` is an English referring expression that matches NOTHING in its photo. Write zh: a natural colloquial "
        "Chinese translation (keep all cues, add nothing), and para: an English paraphrase with the same meaning and different "
        "words. Fields: gid, zh, para, ok."),
    "absent_func": (
        "The category `absent_cat` does NOT appear in the photo; `others` lists what IS in it. Write a functional / affordance "
        "query that fits `absent_cat` without naming it, a synonym or a hypernym, and that would NOT plausibly be answered by "
        "any object in `others` (no 'something to sit on' if any seat or bench or bed is present). func_en: English, 2-8 words; "
        "func_zh: Chinese. ok=false if impossible. Fields: gid, func_en, func_zh, ok."),
}
FIELDS = {"present": ("gid", "orig", "cat", "others"), "unique": ("gid", "cat", "supercat", "others"),
          "absent": ("gid", "orig"), "absent_func": ("gid", "absent_cat", "others")}

cache_fn = f"{a.out}/llm_cache.jsonl"
cache = {}
if os.path.exists(cache_fn):
    for l in open(cache_fn):
        d = json.loads(l); cache[d["gid"]] = d
lock = threading.Lock()


def work(batch):
    kind = batch[0]["kind"]
    user = json.dumps([{k: g[k] for k in FIELDS[kind]} for g in batch], ensure_ascii=False)
    for _ in range(3):
        try:
            res = chat(SYS + TASK[kind], user)
            res = res.get("results", res) if isinstance(res, dict) else res
            got = [r for r in res if isinstance(r, dict) and r.get("gid") in {g["gid"] for g in batch}]
            if len(got) >= 0.8 * len(batch):
                with lock, open(cache_fn, "a") as f:
                    for r in got:
                        cache[r["gid"]] = r; f.write(json.dumps(r, ensure_ascii=False) + "\n")
                return
        except Exception as e:
            print("llm fail", kind, str(e)[:100], flush=True)


todo = [g for g in groups if g["gid"] not in cache]
batches = []
for kind in TASK:
    v = [g for g in todo if g["kind"] == kind]
    batches += [v[i:i + 15] for i in range(0, len(v), 15)]
with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(work, batches))

# ---------------- assemble
zh = lambda s: bool(re.search(r"[一-鿿]", s or ""))
VAR = {"present": [("orig", "orig", "en"), ("zh", "zh", None), ("para", "para", "en"), ("func", "func", None),
                   ("vague", "vague", "en"), ("zh_vague", "zh_vague", "zh")],
       "unique": [("func_only", "func_en", "en"), ("func_only", "func_zh", "zh"), ("generic", "generic_en", "en"),
                  ("generic", "generic_zh", "zh")],
       "absent": [("orig", "orig", "en"), ("zh", "zh", None), ("para", "para", "en")],
       "absent_func": [("func_only", "func_en", "en"), ("func_only", "func_zh", "zh")]}
rows, drop = [], collections.Counter()
for g in groups:
    c = cache.get(g["gid"])
    if c is None or c.get("ok") is False:
        drop[f"{g['kind']}/no_llm"] += 1; continue
    for shift, field, lang in VAR[g["kind"]]:
        t = g["orig"] if field == "orig" else str(c.get(field) or "").strip()
        if not t:
            drop[f"{g['kind']}/{shift}/empty"] += 1; continue
        cname = g.get("cat") or g.get("absent_cat")
        if shift in ("func", "func_only", "generic") and cname and cname != "person" and re.search(r"\b" + re.escape(cname) + r"s?\b", t.lower()):
            drop[f"{g['kind']}/{shift}/names_cat"] += 1; continue
        if shift == "para" and re.sub(r"\W+", " ", t.lower()).strip() == re.sub(r"\W+", " ", g["orig"].lower()).strip():
            drop["para_identical"] += 1; continue
        rows.append(dict(gid=g["gid"], kind=g["kind"], shift=shift, lang=lang or ("zh" if zh(t) else "en"), text=t,
                         present=g["box"] is not None, box=g["box"], image_id=g["image_id"], file=g["file"], W=g["W"], H=g["H"],
                         src=g["src"], cat=g.get("cat"), orig=g["orig"]))
with open(f"{a.out}/groups.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
by = collections.OrderedDict()
for r in rows:
    rec = by.setdefault(r["image_id"], dict(image_id=r["image_id"], file=r["file"], W=r["W"], H=r["H"], questions=[]))
    rec["questions"].append(dict(text=r["text"], present=r["present"], box=r["box"], n_targets=int(r["present"]),
                                 boxes=[r["box"]] if r["present"] else [], meta=dict(gid=r["gid"], kind=r["kind"], shift=r["shift"], lang=r["lang"])))
with open(f"{a.out}/eval_rg.jsonl", "w") as f:
    for rec in by.values():
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
st = dict(drop=dict(drop), rows=len(rows), images=len(by),
          by=dict(collections.Counter(f"{r['kind']}/{r['shift']}/{r['lang']}" for r in rows)))
json.dump(st, open(f"{a.out}/stats.json", "w"), indent=1)
print(json.dumps(st, indent=1))
