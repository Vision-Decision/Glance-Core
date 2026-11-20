"""v1.2 category-free grounding data, step 3 (rules): filter the LLM wording, keep only blind-verified items, add the
consistency anchors, write training records.

Kept iff (all rules; counts of every drop reason go to stats):
  language     zh text has Han characters, en text has none; length 1-14 words (en) / 2-24 characters (zh)
  no leak      positives: no name / synonym / plural of the target's category (scripts/robust/cat_names.py + ground_v2
               lexicon) and, for function / generic, of no COCO category at all; negatives: no COCO category name except the
               relation anchor's; relation queries must name their anchor; no digits, "box", "label", 框, 编号, 号
  verified     the blind verifier's answer is exactly [target] (positives) or "none" (negatives)
  unique       a text appears once per image (case-insensitive)

Each positive gets a consistency group with an *anchor* phrasing written by rule from the annotation ("the cup", or
"the cat to the left of the refrigerator" when the category repeats): {"grp": "<image>:<obj>", "anchor": true} on the anchor,
{"grp": ...} on the variants. Training applies the consistency loss variant -> stop-grad(anchor) (train.py consistency_loss).

  python scripts/v12/catfree_build.py --scenes scenes.jsonl --gen gen.jsonl --out_dir out/ [--n_pos 150000 --n_neg 50000]
"""
import argparse, collections, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ground_v2"))
sys.path.insert(0, os.path.join(HERE, "..", "robust"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from catfree_scenes import REL_EN, REL_ZH  # noqa: E402
from lexicon import EN, ZH, ZH_MW, IS_ZH, mentions, names_of  # noqa: E402
from cat_names import NAMES as RNAMES  # noqa: E402

BAD = re.compile(r"[0-9０-９]|\b(box|boxes|label|labeled|labelled|numbered|number)\b|框|编号|号", re.I)
QTYPE = {"func_pos": "cf_func", "generic_pos": "cf_generic", "rel_pos": "cf_rel",
         "func_neg": "cf_func_neg", "generic_neg": "cf_generic_neg", "rel_neg": "cf_rel_neg", "pos_absent": "cf_pos_absent"}
SUB_QTYPE = {"rel_absent": "cf_rel_neg_absent", "rel_thing": "cf_rel_neg_thing"}


def cat_words(c):
    return list(dict.fromkeys(RNAMES.get(c, []) + names_of(c)))


ALL_CATS = list(EN)


def anchor_text(sc, o, cnt):
    """rule-made plain description of object o (English): the category name if unique, else category + relation."""
    if cnt[o["cat"]] == 1:
        return f"the {EN[o['cat']]}"
    from catfree_scenes import rel_holds
    for B in sc["objs"]:
        if B["id"] == o["id"] or cnt[B["cat"]] != 1 or B["area"] < 0.02:
            continue
        for r in ("left", "right", "above", "below"):
            same = [x for x in sc["objs"] if x["cat"] == o["cat"]]
            if rel_holds(r, o["box"], B["box"]) and not any(
                    x["id"] != o["id"] and rel_holds(r, x["box"], B["box"], relaxed=True) for x in same):
                return f"the {EN[o['cat']]} {REL_EN[r]} the {EN[B['cat']]}"
    return None


def check(text, t, sc, obj):
    lang = t["lang"]
    if not text or BAD.search(text):
        return "bad_token"
    zh = bool(IS_ZH.search(text))
    if zh != (lang == "zh"):
        return "language"
    if (lang == "en" and not 1 <= len(text.split()) <= 14) or (lang == "zh" and not 2 <= len(text) <= 24):
        return "length"
    k = t["kind"]
    anchor_cat = obj[t["anchor"]]["cat"] if "anchor" in t else None
    if k == "pos_absent":  # names the absent category on purpose; must name it
        anchor_cat = t["absent_cat"]
    allowed = set(w.lower() for w in cat_words(anchor_cat)) if anchor_cat else set()
    if k.endswith("_pos") and mentions(text, cat_words(obj[t["target"]]["cat"])):
        return "leak_target"
    # no other COCO category may be named either (except words of the relation anchor; person words are allowed: "the
    # thing the man is holding" / "让人坐的东西" stay category-free for the target)
    for c in ALL_CATS:
        if c in ("person", anchor_cat):
            continue
        for w in cat_words(c):
            if w.lower() not in allowed and mentions(text, [w]):
                return "leak_other_category"
    if anchor_cat and not mentions(text, cat_words(anchor_cat)):
        return "anchor_not_named"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pos", type=int, default=150000)
    ap.add_argument("--n_neg", type=int, default=50000)
    ap.add_argument("--max_q", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    from dv.data.prep_v1 import plen_image
    scenes = {}
    for l in (l for fn in a.scenes.split(",") for l in open(fn)):
        s = json.loads(l); scenes[s["image_id"]] = s
    st = collections.defaultdict(collections.Counter)
    per_img, done = [], set()
    for l in (l for fn in a.gen.split(",") for l in open(fn)):
        g = json.loads(l)
        if g["image_id"] in done:  # generation shards may overlap at their boundary
            continue
        done.add(g["image_id"])
        sc = scenes[g["image_id"]]
        obj = {o["id"]: o for o in sc["objs"]}
        tasks = {t["tid"]: t for t in sc["tasks"]}
        cnt = collections.Counter(o["cat"] for o in sc["objs"])
        ans = {tuple(v): g["verify"].get(q) for q, v in g["qmap"].items()}
        seen, kept = set(), []
        for tid, texts in g["gen"].items():
            t = tasks[tid]
            for i, text in enumerate(texts):
                k = t["kind"]
                st["raw"][k] += 1
                why = check(text, t, sc, obj)
                if why is None and text.lower() in seen:
                    why = "duplicate"
                if why is None:
                    v = ans.get((tid, i))
                    if k.endswith("_pos"):
                        ok = isinstance(v, list) and [int(x) for x in v if str(x).isdigit()] == [t["target"]] and len(v) == 1
                    else:
                        ok = v == "none" or v == ["none"]
                    if not ok:
                        why = "verify_missing" if v is None else "verify_" + ("none" if v in ("none", ["none"]) else
                                                                                "other" if v in ("other", ["other"]) else "ids")
                if why:
                    st["drop"][f"{k}/{why}"] += 1
                    continue
                seen.add(text.lower())
                kept.append((t, text))
                st["kept"][k] += 1
                st["kept"][f"{k}/{t['lang']}"] += 1
        if kept:
            per_img.append((sc, kept, cnt, obj))
    # ---- sample to the requested totals (all kinds proportionally), then build records with anchors
    pos = [(i, j) for i, (_, k, _, _) in enumerate(per_img) for j, (t, _) in enumerate(k) if t["kind"].endswith("_pos")]
    neg = [(i, j) for i, (_, k, _, _) in enumerate(per_img) for j, (t, _) in enumerate(k) if not t["kind"].endswith("_pos")]
    rng.shuffle(pos); rng.shuffle(neg)
    st["out"]["pos_available"] = len(pos); st["out"]["neg_available"] = len(neg)
    pos = pos[: a.n_pos]
    # negatives per family (func / generic / rel) in proportion to that family's positives, so every category-free
    # positive type has an absent counterpart of the same type (S3 sprint: otherwise false-absent becomes false-present)
    fam = lambda ij: per_img[ij[0]][1][ij[1]][0]["kind"].split("_")[0]
    npos = collections.Counter(fam(x) for x in pos)
    negs = collections.defaultdict(list)
    for x in neg:
        negs[fam(x)].append(x)
    want = {f: round(a.n_neg * npos[f] / max(1, len(pos))) for f in npos}
    sel = []
    for f, w in want.items():
        sel += negs[f][:w]
    short = a.n_neg - len(sel)  # a family without enough negatives: fill from the others
    if short > 0:
        rest = [x for f in negs for x in negs[f][want.get(f, 0):]]
        sel += rest[:short]
    keep = set(pos) | set(sel)
    st["out"]["neg_by_family"] = dict(collections.Counter(fam(x) for x in sel))
    st["out"]["pos_by_family"] = dict(npos)
    os.makedirs(a.out_dir, exist_ok=True)
    n_rec = 0
    with open(f"{a.out_dir}/catfree.jsonl", "w") as f:
        for i, (sc, kept, cnt, obj) in enumerate(per_img):
            qs, anchors = [], {}
            for j, (t, text) in enumerate(kept):
                if (i, j) not in keep:
                    continue
                k = t["kind"]
                q = dict(type="box", text=text, qtype=SUB_QTYPE.get(t.get("sub"), QTYPE[k]), lang=t["lang"], src="catfree")
                if "anchor" in t:  # relation anchor, for spot-check rendering
                    q["ref_box"], q["ref_cat"] = obj[t["anchor"]]["box"], obj[t["anchor"]]["cat"]
                if k.endswith("_pos"):
                    o = obj[t["target"]]
                    q.update(present=True, box=o["box"], boxes=[o["box"]], n_targets=1, neg=None)
                    cg = f"{sc['image_id']}:{o['id']}"
                    if cg not in anchors:
                        at = anchor_text(sc, o, cnt)
                        anchors[cg] = at
                        if at:
                            qs.append(dict(type="box", text=at, qtype="cf_anchor", lang="en", src="catfree_rule", present=True,
                                           box=o["box"], boxes=[o["box"]], n_targets=1, neg=None, grp=cg, anchor=True))
                            st["kept"]["cf_anchor"] += 1
                    if anchors[cg]:
                        q["grp"] = cg
                        st["consistency"]["variants_with_anchor"] += 1
                    else:
                        st["consistency"]["variants_without_anchor"] += 1
                else:
                    q.update(present=False, box=None, boxes=[], n_targets=0, neg="hard")
                qs.append(q)
            if not qs:
                continue
            base = dict(src="catfree", img=f"coco/train2014/{sc['file']}", wh=[sc["W"], sc["H"]], max_vis=640,
                        plen=plen_image(sc["W"], sc["H"], 640), image_id=sc["image_id"])
            # images rarely exceed max_q; if they do, split by consistency group so a group stays in one record
            groups = collections.OrderedDict()
            for q in qs:
                groups.setdefault(q.get("grp") or id(q), []).append(q)
            cur = []
            for gq in groups.values():
                if cur and len(cur) + len(gq) > a.max_q:
                    f.write(json.dumps(dict(base, questions=cur), ensure_ascii=False) + "\n"); n_rec += 1; cur = []
                cur += gq
            if cur:
                f.write(json.dumps(dict(base, questions=cur), ensure_ascii=False) + "\n"); n_rec += 1
    st["out"]["records"] = n_rec
    st["out"]["images"] = len({sc["image_id"] for sc, *_ in per_img})
    st["out"]["pos_written"] = len(pos); st["out"]["neg_written"] = len(sel)
    out = {k: dict(sorted(v.items())) for k, v in st.items()}
    json.dump(out, open(f"{a.out_dir}/stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
