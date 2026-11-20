"""Module 3 (rules): turn scenes (Module 1) + LLM sentences (Module 2) into box questions with exact answers.

Every answer comes from COCO annotations: boxes, counts and category presence are exact; LLM sentences are kept only
when they pass the rule filters below AND the independent verification call (gen_llm.py) agrees with the intended
target set (positives) or says the sentence fits nothing (negatives).

Question record (dv/data/mixed.py box block, plus the multi-box fields agreed with the multi-box head work):
  {"type": "box", "text", "present", "box": [x1,y1,x2,y2] px | null (null unless exactly one target),
   "boxes": [[x1,y1,x2,y2], ...] (0 = absent, 1 = single, >=2 = multi), "n_targets", "qtype", "lang", "neg", "src"}

qtypes
  positives  pos_short pos_mid pos_long pos_colloq pos_zh pos_func  (LLM, one target)
             pos_multi        (LLM group phrase naming every target)      pos_multi_human (gRefCOCO multi-target, train)
             pos_cat          (bare category, the only instance)         pos_cat_all (bare category / plural -> all instances)
             pos_all          ("all the X", "every X", "both X", "the three X", "所有的X")
             pos_pos_sole     (position word / superlative on the only instance: "the leftmost cat" with one cat)
             pos_pos_multi    (leftmost / rightmost / biggest among several, clear geometric margin)
             pos_human        (RefCOCO/+/g train expression)
  negatives  neg_colour neg_action (LLM, verified)   neg_count ("the six zebras" with three; "two cats" with one)
             neg_similar (similar category absent: noun swap of the verified medium phrase, or bare)
             neg_absent (category absent)   neg_position (sole object, clearly on the other side)
             neg_size ("the tiny X" for an object filling >=30% of the photo)   neg_human (gRefCOCO no-target, train)

Filters (the deck's three plus ours): category-only / count / position questions skipped when any instance of the
category is crowd, tiny or mostly cut off at the image edge (Module 1 `marginal`), or n > 8; multi-target sentences
must mention every target (category word, and a number word for same-category groups); short phrases must be unique
in the image (not a bare category name when the category has several instances; not shared with another object);
function phrasings must not name the category / synonyms; no mention of drawn boxes or numbers; language check.

  python scripts/ground_v2/build_questions.py --scenes scenes.jsonl --gen gen.jsonl --out_dir out/ [--test]
"""
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lexicon import EN, EN_PL, ZH, ZH_MW, NUM, ZH_NUM, SIMILAR, IS_ZH, mentions, names_of, names_for_mention  # noqa: E402
from dv.data.prep_v1 import plen_image  # noqa: E402
from build_scenes import describable  # noqa: E402

FIELD_QT = {"short": "pos_short", "medium": "pos_mid", "long": "pos_long", "colloquial": "pos_colloq", "zh_short": "pos_zh",
            "zh": "pos_zh", "function": "pos_func", "zh_function": "pos_func", "group_en": "pos_multi", "group_zh": "pos_multi",
            "colour_neg": "neg_colour", "action_neg": "neg_action"}
# references to the drawn marks only ("box truck", "pizza box", "bottle label", "眼镜框" are fine)
BAD = re.compile(r"\b(numbered|(?:box|label|object|item|mark) ?#?\d+|#\d+)\b|编号|标号|\d+号(?:框|物体)", re.I)
NUMWORD = re.compile(r"\b(two|both|pair|three|four|couple)\b|两|二|俩|一对|三|四", re.I)
FUNC_CUE = re.compile(r"\b(to|for|you|you'd|use|used|using|can|could|would|helps?|lets?|keeps?|holds?|holding|carr(?:y|ies|ying)|that|which|where"
                      r"|rests?|resting|sits?|set)\b|用|能|可以|来|供|装|放|坐|吃|喝|穿|骑|种|摆|挂|背|切|踩|绑|插|盛|躺|睡", re.I)
ARTICLES = re.compile(r"^(the|a|an|that|this)\s+", re.I)


def Q(text, present, boxes, qtype, lang, neg=None, src="ground_v2"):
    boxes = [[round(float(v), 2) for v in b] for b in boxes]
    return dict(type="box", text=text, present=present, box=boxes[0] if len(boxes) == 1 else None, boxes=boxes,
                n_targets=len(boxes), qtype=qtype, lang=lang, neg=neg, src=src)


PILE = {"book", "banana", "carrot", "broccoli", "apple", "orange", "donut"}
ANIMATE = {"person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}


def overlap_small(a, b):
    """intersection / area of the smaller box"""
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def cxcy(o, W, H):
    b = o["box"]
    return (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H


def lang_ok(text, zh):
    has = bool(IS_ZH.search(text))
    return has if zh else not has


ACT_STOP = {"the", "a", "an", "on", "in", "at", "of", "with", "and", "its", "his", "her", "their", "to", "from", "into", "being",
            "down", "up", "near", "by", "over", "while", "something"}


def kept_action(neg, action):
    """True if the wrong-action phrase still contains a content word of the observed action (stacked actions)"""
    if not isinstance(action, str):
        return False
    ws = {w for w in re.findall(r"[a-z]+", action.lower())
          if w not in ACT_STOP and (w.endswith("ing") or w.endswith("ed") or w in {"asleep", "seated", "curled"}) and w not in {"red"}}
    return bool(ws & set(re.findall(r"[a-z]+", neg.lower())))


def swap_noun(text, cat, new):
    """replace the first mention of cat (or a synonym) with `new`; fix a/an"""
    for w in sorted([x for x in names_of(cat) if not IS_ZH.search(x)], key=len, reverse=True):
        m = re.search(r"\b" + re.escape(w.lower()) + r"\b", text.lower())
        if m:
            t = text[:m.start()] + new + text[m.end():]
            t = re.sub(r"\b(a|an)\s+(?=" + re.escape(new) + r")", "an " if new[0].lower() in "aeiou" else "a ", t, flags=re.I)
            return t
    return None


def scene_questions(sc, gen, rng, stats, test=False):
    W, H = sc["W"], sc["H"]
    objs = sc["objects"]
    by_ann = {o["ann_id"]: o for o in objs}
    by_num = {o["num"]: o for o in objs if o.get("num")}
    cnt = collections.Counter(o["cat"] for o in objs)
    inst = collections.defaultdict(list)
    for o in objs:
        inst[o["cat"]].append(o)
    # category / count / position questions need a complete, unambiguous instance set (spot check round 1: piles of
    # books / carrots are under-annotated, nested duplicate boxes make "all the X" ill-defined)
    reliable = {c for c, l in inst.items() if len(l) <= 8 and c not in PILE and not any(o["crowd"] or o["marginal"] for o in l)
                and not any(overlap_small(a["box"], b["box"]) > 0.6 for i, a in enumerate(l) for b in l[i + 1:])}
    qs = []

    # ---------------- A. LLM sentences (verified)
    if gen is not None:
        cands = gen.get("cands")
        if cands is None:  # --no_verify run
            from gen_llm import candidates
            cands = [dict(field=c[0], text=c[1], kind=c[2], intended=c[3], owner=c[4], verified=c[3]) for c in candidates(sc, gen["gen"])]
        shorts = collections.Counter(c["text"].lower() for c in cands if c["field"] in ("short", "zh_short"))
        mediums = {c["owner"]: c for c in cands if c["field"] == "medium"}
        actions = {o.get("id"): o.get("action") for o in gen.get("gen", {}).get("objects", []) if isinstance(o, dict)}
        for c in cands:
            f, t = c["field"], " ".join(c["text"].split())
            qt = FIELD_QT[f]
            stats["llm_cand"][qt] += 1
            zh = f.startswith("zh") or f == "group_zh"
            reason = None
            if any(not describable(by_num[n], objs, cnt) for n in ([c["owner"]] if isinstance(c["owner"], int) else c["intended"])):
                reason = "target_not_describable"  # scenes built before the rule existed (round-3 spot check)
            elif not lang_ok(t, zh):
                reason = "lang"
            elif BAD.search(t):
                reason = "mentions_box"
            elif len(t) > (60 if zh else 200) or len(t) < 2:
                reason = "length"
            elif f in ("function", "zh_function") and mentions(t, names_of(by_num[c["owner"]]["cat"])):
                reason = "func_names_category"
            elif f in ("function", "zh_function") and (by_num[c["owner"]]["cat"] in ANIMATE or not FUNC_CUE.search(t)):
                reason = "func_no_function_cue"  # "the long blue thing on the sand" is a vague phrase, not a use
            elif f in ("short", "zh_short"):
                o = by_num[c["owner"]]
                core = ARTICLES.sub("", t.lower()).strip(" .")
                if cnt[o["cat"]] > 1 and (core in [x.lower() for x in names_of(o["cat"])] or core.rstrip("s") in [x.lower() for x in names_of(o["cat"])]):
                    reason = "short_not_unique_bare_category"
                elif shorts[t.lower()] > 1:
                    reason = "short_shared"
            elif f.startswith("group"):
                grp = sc["groups"][int(c["owner"][1:]) - 1]
                gcats = [by_num[n]["cat"] for n in grp["nums"]]
                if not all(mentions(t, names_for_mention(gc)) for gc in set(gcats)):
                    reason = "group_missing_target"
                elif grp["kind"] == "same_cat" and not NUMWORD.search(t):
                    reason = "group_no_number_word"
            elif f == "action_neg" and by_num[c["owner"]]["cat"] not in ANIMATE:
                reason = "action_neg_inanimate"  # "the bus pulling away": motion cannot be judged in a still
            elif f == "action_neg" and kept_action(t, actions.get(c["owner"])):
                reason = "action_neg_keeps_true_action"  # "dark dog curled up running": stacked, self-contradictory
            elif f in ("colour_neg", "action_neg"):
                m = mediums.get(c["owner"])
                if m and m["text"].strip().lower() == t.lower():
                    reason = "neg_identical"
            if reason is None:
                v = c.get("verified")
                if c["kind"] == "pos":
                    if not (isinstance(v, list) and sorted(int(x) for x in v if str(x).isdigit()) == sorted(c["intended"])):
                        reason = "verify_mismatch"
                else:
                    if v != []:
                        reason = "verify_not_empty"
            if reason:
                stats["llm_drop"][f"{qt}/{reason}"] += 1
                continue
            stats["llm_keep"][qt] += 1
            if c["kind"] == "pos":
                tgt = [by_num[n]["box"] for n in c["intended"]]
                qs.append(Q(t, True, tgt, qt, "zh" if zh else "en"))
            else:
                qs.append(Q(t, False, [], qt, "en", neg="hard"))
        # similar-category noun swap of the verified medium phrase
        for owner, m in mediums.items():
            o = by_num[owner]
            if not any(q["text"] == " ".join(m["text"].split()) for q in qs):
                continue
            sims = [s for s in SIMILAR.get(o["cat"], []) if cnt[s] == 0]
            if not sims:
                continue
            s = rng.choice(sims)
            t = swap_noun(" ".join(m["text"].split()), o["cat"], EN[s])
            if t:
                qs.append(Q(t, False, [], "neg_similar", "en", neg="hard"))

    # ---------------- B. category / count / position questions from annotations
    for c, l in inst.items():
        if c not in reliable:
            continue
        n = len(l); en, pl, zh, mw = EN[c], EN_PL[c], ZH[c], ZH_MW[c]
        boxes = [o["box"] for o in l]
        if n == 1:
            o = l[0]; cx, cy = cxcy(o, W, H)
            qs += [Q(rng.choice([en, f"the {en}"]), True, boxes, "pos_cat", "en"), Q(zh, True, boxes, "pos_cat", "zh")]
            opts = [f"the leftmost {en}", f"the rightmost {en}", f"the biggest {en}", f"the smallest {en}", f"the only {en}",
                    f"the {en} closest to the camera", f"the largest {en}"]
            if cx < 0.4: opts += [f"the {en} on the left", f"the {en} on the left side"]
            if cx > 0.6: opts += [f"the {en} on the right", f"the {en} on the right side"]
            if cy < 0.35: opts += [f"the {en} at the top"]
            if cy > 0.65: opts += [f"the {en} at the bottom"]
            for t in rng.sample(opts, 2):
                qs.append(Q(t, True, boxes, "pos_pos_sole", "en"))
            zopts = [f"最左边的{zh}", f"最右边的{zh}", f"最大的{zh}", f"最小的{zh}", f"唯一的那{mw}{zh}"]
            if cx < 0.4: zopts.append(f"左边的{zh}")
            if cx > 0.6: zopts.append(f"右边的{zh}")
            qs.append(Q(rng.choice(zopts), True, boxes, "pos_pos_sole", "zh"))
            k = rng.choice([3, 4])  # >= n+2: a second, unannotated instance must not turn the negative true
            qs.append(Q(rng.choice([f"the {NUM[k]} {pl}", f"{NUM[k]} {pl}"]), False, [], "neg_count", "en", neg="hard"))
            if rng.random() < 0.5:
                qs.append(Q(f"{ZH_NUM[k]}{mw}{zh}", False, [], "neg_count", "zh", neg="hard"))
            b = o["box"]
            if cx < 0.3 and b[2] < 0.5 * W:
                qs.append(Q(rng.choice([f"the {en} on the right", f"the {en} on the right side"]), False, [], "neg_position", "en", neg="hard"))
            elif cx > 0.7 and b[0] > 0.5 * W:
                qs.append(Q(rng.choice([f"the {en} on the left", f"the {en} on the left side"]), False, [], "neg_position", "en", neg="hard"))
            if o["area"] >= 0.3:
                qs.append(Q(rng.choice([f"the tiny {en}", f"the very small {en}"]), False, [], "neg_size", "en", neg="hard"))
        else:
            qs += [Q(rng.choice([en, pl]), True, boxes, "pos_cat_all", "en"), Q(zh, True, boxes, "pos_cat_all", "zh")]
            opts = [f"all the {pl}", f"every {en}", f"all {pl}", f"the {NUM[n]} {pl}", f"all {NUM[n]} {pl}"]
            if n == 2: opts += [f"both {pl}", f"both of the {pl}"]
            for t in rng.sample(opts, 2):
                qs.append(Q(t, True, boxes, "pos_all", "en"))
            qs.append(Q(rng.choice([f"所有的{zh}", f"全部{zh}", f"这{ZH_NUM[n]}{mw}{zh}"]), True, boxes, "pos_all", "zh"))
            k = rng.choice([x for x in range(n + 2, n + 5) if x <= 12])
            qs.append(Q(rng.choice([f"the {NUM[k]} {pl}", f"{NUM[k]} {pl}", f"all {NUM[k]} {pl}"]), False, [], "neg_count", "en", neg="hard"))
            if rng.random() < 0.5:
                qs.append(Q(f"{ZH_NUM[k]}{mw}{zh}", False, [], "neg_count", "zh", neg="hard"))
            srt = sorted(l, key=lambda o: cxcy(o, W, H)[0])
            c0, c1 = cxcy(srt[0], W, H)[0], cxcy(srt[1], W, H)[0]
            cl, cl1 = cxcy(srt[-1], W, H)[0], cxcy(srt[-2], W, H)[0]
            if c1 - c0 > 0.12 and srt[0]["box"][2] <= srt[1]["box"][0] + 0.05 * W:
                qs.append(Q(rng.choice([f"the leftmost {en}", f"the {en} on the far left", f"the {en} furthest to the left"]), True, [srt[0]["box"]], "pos_pos_multi", "en"))
            if cl - cl1 > 0.12 and srt[-1]["box"][0] >= srt[-2]["box"][2] - 0.05 * W:
                qs.append(Q(rng.choice([f"the rightmost {en}", f"the {en} on the far right", f"最右边的{zh}"]), True, [srt[-1]["box"]],
                            "pos_pos_multi", "en"))
            ar = sorted(l, key=lambda o: -o["area"])
            if ar[0]["area"] >= 2.5 * ar[1]["area"]:
                qs.append(Q(rng.choice([f"the biggest {en}", f"the largest {en}", f"最大的{zh}"]), True, [ar[0]["box"]], "pos_pos_multi", "en"))
    for q in qs:  # zh texts produced by mixed rng.choice above
        if IS_ZH.search(q["text"]):
            q["lang"] = "zh"
    # similar / absent categories
    present = set(cnt)
    for c in present:
        sims = [s for s in SIMILAR.get(c, []) if s not in present]
        if sims and rng.random() < 0.6:
            s = rng.choice(sims)
            qs.append(Q(rng.choice([f"the {EN[s]}", EN[s], ZH[s]]), False, [], "neg_similar", "en", neg="hard"))
    far = [c for c in EN if c not in present and c not in sum((SIMILAR.get(p, []) for p in present), [])]
    for c in rng.sample(far, min(2, len(far))):
        qs.append(Q(rng.choice([f"the {EN[c]}", EN_PL[c], ZH[c]]), False, [], "neg_absent", "en", neg="easy"))
    # ---------------- C. existing human annotations (train only)
    hum = [(t, o) for o in objs if not o["crowd"] for t in o["refs"][:2]]
    for t, o in rng.sample(hum, min(12, len(hum))):
        qs.append(Q(t, True, [o["box"]], "pos_human", "en", src="refcoco"))
    gm = [g for g in sc.get("g_multi", []) if all(a in by_ann for a in g["ann_ids"])]
    for g in rng.sample(gm, min(4, len(gm))):
        qs.append(Q(g["text"], True, [by_ann[a]["box"] for a in g["ann_ids"]], "pos_multi_human", "en", src="grefcoco"))
    for t in rng.sample(sc.get("g_none", []), min(3, len(sc.get("g_none", [])))):
        qs.append(Q(t, False, [], "neg_human", "en", neg="hard", src="grefcoco"))
    for q in qs:
        q["lang"] = "zh" if IS_ZH.search(q["text"]) else "en"
    # ---------------- dedupe; drop texts that would get two different answers
    by_text = collections.defaultdict(list)
    for q in qs:
        by_text[q["text"].lower().strip(" .")].append(q)
    out = []
    for k, l in by_text.items():
        ans = {(q["present"], tuple(map(tuple, sorted(q["boxes"])))) for q in l}
        if len(ans) > 1:
            stats["conflict_drop"][l[0]["qtype"]] += len(l)
            continue
        out.append(l[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--gen", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test", action="store_true", help="held-out test set: one record per image, LLM-free types kept")
    ap.add_argument("--require_gen", action="store_true", help="skip scenes without an LLM generation")
    ap.add_argument("--max_q", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    gens = {}
    for p in filter(None, a.gen.split(",")):
        for l in open(p):
            g = json.loads(l)
            gens[g["image_id"]] = g
    stats = collections.defaultdict(collections.Counter)
    n_img, n_rec = 0, 0
    usage = collections.Counter()
    name = "test" if a.test else "train"
    with open(f"{a.out_dir}/{name}.jsonl", "w") as f:
        for l in open(a.scenes):
            sc = json.loads(l)
            g = gens.get(sc["image_id"])
            if g is None and a.require_gen:
                continue
            if g:
                for p, o in g.get("usage", []):
                    usage["prompt"] += p; usage["out"] += o; usage["calls"] += 1
            qs = scene_questions(sc, g, rng, stats, a.test)
            if not qs:
                continue
            n_img += 1
            for q in qs:
                stats["qtype"][q["qtype"]] += 1
                stats["lang"][q["lang"]] += 1
                stats["presence"]["present" if q["present"] else "absent"] += 1
                stats["n_targets"][min(q["n_targets"], 4)] += 1
            base = dict(src="ground_v2", img=f"coco/{sc['year']}/{sc['file']}", wh=[sc["W"], sc["H"]], max_vis=640,
                        plen=plen_image(sc["W"], sc["H"], 640), image_id=sc["image_id"])
            if a.test:
                f.write(json.dumps(dict(base, url=sc["url"], license=sc["license"], questions=qs), ensure_ascii=False) + "\n"); n_rec += 1
            else:
                rng.shuffle(qs)
                for i in range(0, len(qs), a.max_q):
                    f.write(json.dumps(dict(base, questions=qs[i:i + a.max_q]), ensure_ascii=False) + "\n"); n_rec += 1
    tot = sum(stats["qtype"].values())
    st = dict(images=n_img, records=n_rec, questions=tot, q_per_image=round(tot / max(1, n_img), 1),
              llm_usage=dict(usage), **{k: dict(sorted(v.items(), key=lambda x: str(x[0]))) for k, v in stats.items()})
    st["llm_keep_rate"] = {k: round(st.get("llm_keep", {}).get(k, 0) / v, 3) for k, v in st.get("llm_cand", {}).items()}
    json.dump(st, open(f"{a.out_dir}/stats_{name}.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(st, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
