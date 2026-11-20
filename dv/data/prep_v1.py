"""v1 data preparation (typed decision mixture). Runs on the Colab VM after scripts/v1/fetch.sh.

  python -m dv.data.prep_v1 natural   # RefCOCO/+/g + gRefCOCO + hard negatives      -> {out}/pool_natural.jsonl, calib_natural.jsonl
  python -m dv.data.prep_v1 evalhash  # perceptual hashes of every eval image we report -> {out}/eval_hashes.json
  python -m dv.data.prep_v1 gui       # GroundCUA / OS-Atlas / AGUVIS (hash-deduped)    -> {out}/pool_gui.jsonl, calib_gui.jsonl
  python -m dv.data.prep_v1 pixmo     # pixmo-points / pixmo-count (images downloaded)  -> {out}/pool_pixmo.jsonl
  python -m dv.data.prep_v1 text      # typed text decisions (license + contamination filtered) -> {out}/pool_text.jsonl, calib_text.jsonl
  python -m dv.data.prep_v1 mix       # sample pools to the target mixture               -> {out}/train.jsonl, stats.json

Record format: see dv/data/mixed.py. Every step writes {out}/stats_<step>.json.
"""
import argparse, collections, glob, hashlib, io, json, math, os, random, re, sys

COLORS = ["white", "black", "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown", "gray", "grey",
          "tan", "beige", "silver", "gold", "dark", "light"]
COLOR_EQ = {"gray": "grey", "grey": "gray"}
SWAP_COLORS = ["white", "black", "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown"]
ATTR = set(COLORS) | {"striped", "stripes", "plaid", "checkered", "dotted", "spotted", "shirt", "jacket", "coat", "hat",
                      "cap", "dress", "skirt", "shorts", "jeans", "pants", "sweater", "hoodie", "vest", "tie", "glasses",
                      "helmet", "scarf", "uniform", "suit", "wooden", "metal", "plastic", "glass", "leather", "bald",
                      "blond", "blonde", "beard", "umbrella", "bag", "backpack", "phone", "racket", "bat", "glove",
                      "sitting", "standing", "walking", "running", "holding", "eating", "smiling", "laughing", "lying",
                      "sleeping", "skateboard", "surfboard", "kite", "frisbee", "baby", "child", "kid", "boy", "girl",
                      "woman", "lady", "man", "guy", "old", "young", "tall", "short", "small", "big", "large", "little"}
LOC = {"left", "right", "top", "bottom", "middle", "center", "front", "back", "behind", "near", "far", "closest",
       "furthest", "farthest", "first", "second", "third", "upper", "lower", "corner", "side", "foreground", "background"}


def words(s):
    return re.findall(r"[a-z]+", s.lower())


def phash_file(path):
    """64-bit difference hash of the decoded image (robust to re-encoding / resizing) + exact pixel md5."""
    from PIL import Image
    im = Image.open(path).convert("L")
    small = im.resize((9, 8), Image.BILINEAR)
    px = list(small.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | int(px[r * 9 + c] > px[r * 9 + c + 1])
    return f"{bits:016x}", im.size


def resize_dims(W, H, max_tokens, max_upscale, factor=32):  # = dv.data.dataset.resize_dims (no torch import here)
    s = min(math.sqrt(max_tokens * factor * factor / (W * H)), max_upscale)
    while True:
        w = max(factor, int(round(W * s / factor)) * factor)
        h = max(factor, int(round(H * s / factor)) * factor)
        if (w // factor) * (h // factor) <= max_tokens:
            return w, h
        s *= 0.98


def plen_image(W, H, max_vis, max_upscale=2.0):  # noqa: keep signature
    w, h = resize_dims(W, H, max_vis, max_upscale)
    return (w // 32) * (h // 32) + 5


def write_jsonl(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", path, len(recs), "records", sum(len(r["questions"]) for r in recs), "questions", flush=True)


def chunk_record(base, qs, maxq, rng):
    rng.shuffle(qs)
    return [dict(base, questions=qs[i:i + maxq]) for i in range(0, len(qs), maxq)]


# ======================================================================================= natural images
def natural(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prep import load_parquet_refs, norm_text
    rng = random.Random(a.seed)
    coco = json.load(open(f"{a.ann}/annotations/instances_train2014.json"))
    img_info = {im["id"]: im for im in coco["images"]}
    img_cats = collections.defaultdict(set)
    ann_box = {}
    for an in coco["annotations"]:
        img_cats[an["image_id"]].add(an["category_id"])
        x, y, w, h = an["bbox"]
        ann_box[an["id"]] = (an["image_id"], an["category_id"], [x, y, x + w, y + h])
    split_files = {"train": "train", "val": "validation", "testA": "test", "testB": "testB", "test": "test"}
    train_pos, held = [], set()
    for ds in ["refcoco", "refcocoplus", "refcocog"]:
        for sp in (["train", "val", "testA", "testB"] if ds != "refcocog" else ["train", "val", "test"]):
            items = load_parquet_refs(f"{a.ann}/{ds}/{split_files[sp]}.parquet", ds)
            if sp == "train":
                train_pos += items
            else:
                held |= {q["image_id"] for q in items}
    grefs = json.load(open(f"{a.ann}/grefcoco/grefs_unc.json"))
    g_neg, img_vocab_extra = [], collections.defaultdict(set)
    for r in grefs:
        if r["split"] != "train":
            held.add(r["image_id"]); continue
    for r in grefs:
        if r["split"] != "train" or r["image_id"] in held:
            continue
        iid = r["image_id"]
        for s in r["sentences"]:
            img_vocab_extra[iid] |= set(words(s["sent"]))
            if r["no_target"]:
                im = img_info[iid]
                g_neg.append(dict(image_id=iid, file=im["file_name"], W=im["width"], H=im["height"],
                                  text=norm_text(s["sent"]), cat=-1))
    train_pos = [q for q in train_pos if q["image_id"] not in held]
    seen, pos = set(), []
    for q in train_pos:
        k = (q["image_id"], q["text"].lower())
        if k not in seen:
            seen.add(k); pos.append(q)
    gseen, gn = set(), []
    for q in g_neg:
        k = (q["image_id"], q["text"].lower())
        if k not in seen and k not in gseen:
            gseen.add(k); gn.append(q)
    # image vocabulary = all words of all (train) expressions in the image (positive refs + gRefCOCO sents incl. no-target)
    vocab = collections.defaultdict(set)
    for q in pos:
        vocab[q["image_id"]] |= set(words(q["text"]))
    for iid, v in img_vocab_extra.items():
        vocab[iid] |= v
    # gRefCOCO no-target sentences describe things NOT in the image: remove their words from the vocab? No -- keep
    # (conservative: a word seen in any expression of the image is never used as a swapped-in attribute).
    meta = {q["image_id"]: (q["file"], q["W"], q["H"]) for q in pos}
    imgs = sorted(meta)
    rng.shuffle(imgs)
    calib_imgs = set(imgs[: a.calib_images])
    by_img = collections.defaultdict(list)
    for q in pos:
        by_img[q["image_id"]].append(dict(type="box", text=q["text"], present=True, box=q["box"], src=q["src"]))
    for q in gn:
        if q["image_id"] in meta:
            by_img[q["image_id"]].append(dict(type="box", text=q["text"], present=False, box=None, src="grefcoco_nt", neg="hard"))
    existing = {(iid, q["text"].lower()) for iid, l in by_img.items() for q in l}
    n_pos = len(pos)
    n_neg_target = int(n_pos * a.absent_ratio / (1 - a.absent_ratio))
    n_g = sum(1 for l in by_img.values() for q in l if not q["present"])
    n_hard_target = max(int(n_neg_target * a.hard_frac) - n_g, 0)
    # --- hard B: color swap of the image's own expressions
    colorswap = []
    cands = [q for q in pos if sum(w in COLORS for w in words(q["text"])) == 1]
    rng.shuffle(cands)
    for q in cands:
        if len(colorswap) >= n_hard_target // 2:
            break
        ws = words(q["text"])
        c = next(w for w in ws if w in COLORS)
        if c in ("dark", "light", "tan", "beige", "silver", "gold"):
            continue
        opts = [x for x in SWAP_COLORS if x != c and x not in vocab[q["image_id"]] and COLOR_EQ.get(x) not in vocab[q["image_id"]]]
        if not opts:
            continue
        nc = rng.choice(opts)
        t = re.sub(rf"\b{c}\b", nc, q["text"], count=1, flags=re.I)
        k = (q["image_id"], t.lower())
        if k in existing:
            continue
        existing.add(k)
        colorswap.append((q["image_id"], dict(type="box", text=t, present=False, box=None, src="colorswap", neg="hard")))
    # --- hard C: same-category descriptive swap from another image; the donor's attribute words never occur in the
    # target image's expressions
    donors = [q for q in pos if len(words(q["text"])) >= 4 and (set(words(q["text"])) & (ATTR - LOC))]
    by_cat = collections.defaultdict(list)
    for d in donors:
        by_cat[d["cat"]].append(d)
    samecat, tries = [], 0
    n_c = n_hard_target - len(colorswap)
    while len(samecat) < n_c and tries < n_c * 30:
        tries += 1
        iid = rng.choice(imgs)
        cats = [c for c in img_cats[iid] if by_cat.get(c)]
        if not cats:
            continue
        d = rng.choice(by_cat[rng.choice(cats)])
        if d["image_id"] == iid:
            continue
        attrs = set(words(d["text"])) & (ATTR - LOC)
        if attrs & vocab[iid]:
            continue
        k = (iid, d["text"].lower())
        if k in existing:
            continue
        existing.add(k)
        samecat.append((iid, dict(type="box", text=d["text"], present=False, box=None, src="samecat", neg="hard")))
    # --- easy: cross-category swaps (v0 style)
    n_easy = max(0, n_neg_target - n_g - len(colorswap) - len(samecat))
    easy_donors = [q for q in pos if q["src"] == "refcocog" or len(q["text"].split()) >= 3]
    easy, tries = [], 0
    while len(easy) < n_easy and tries < n_easy * 20:
        tries += 1
        iid = rng.choice(imgs); d = rng.choice(easy_donors)
        if d["image_id"] == iid or d["cat"] in img_cats[iid]:
            continue
        k = (iid, d["text"].lower())
        if k in existing:
            continue
        existing.add(k)
        easy.append((iid, dict(type="box", text=d["text"], present=False, box=None, src="swap", neg="easy")))
    for iid, q in colorswap + samecat + easy:
        by_img[iid].append(q)
    train, calib = [], []
    for iid in sorted(by_img):
        if iid not in meta:
            continue
        f, W, H = meta[iid]
        base = dict(src="natural", img=f"coco/train2014/{f}", wh=[W, H], max_vis=640, plen=plen_image(W, H, 640),
                    image_id=iid)
        (calib if iid in calib_imgs else train).extend(chunk_record(base, by_img[iid], 48, rng))
    os.makedirs(a.out, exist_ok=True)
    write_jsonl(f"{a.out}/pool_natural.jsonl", train)
    write_jsonl(f"{a.out}/calib_natural.jsonl", calib)
    cnt = collections.Counter(q["src"] for r in train for q in r["questions"])
    st = dict(images=len({r["image_id"] for r in train}), calib_images=len(calib_imgs), held_out_eval_images=len(held),
              questions=sum(cnt.values()), src=cnt,
              absent=sum(not q["present"] for r in train for q in r["questions"]),
              hard_neg=sum(q.get("neg") == "hard" for r in train for q in r["questions"]),
              easy_neg=sum(q.get("neg") == "easy" for r in train for q in r["questions"]))
    st["absent_ratio"] = st["absent"] / st["questions"]
    st["hard_frac_of_neg"] = st["hard_neg"] / max(1, st["absent"])
    json.dump(st, open(f"{a.out}/stats_natural.json", "w"), indent=1)
    print(json.dumps(st, indent=1))


# ======================================================================================= typed text decisions
# Sources excluded by name: benchmarks with no official train split / Decision Index & JevBench / OpenDecisions
# derivatives (anything from them would be test data), plus BIG-bench canary carriers.
EXCL_SRC = re.compile(r"mmlu.?pro|gpqa|\bhle\b|humanity|bbh|big.?bench|bigbench|musr|sata.?bench|cladder|cruxeval|forecast"
                      r"|pop909|chessbench|bright|toolret|bfcl|api.?bank|routerbench|miniwob|decision.?index|jevbench"
                      r"|opendecisions|jev.?bench|blimp|phish|spam|support.?ticket|humicroedit|bpomp|cfcolor|habermas|nli4ct|isarcasm"
                      r"|\bvast\b|\bacos\b|finentity|\besci\b|contract.?nli|ragtruth|\bhover\b|when2call|\bsgd\b|schema.?guided"
                      r"|home.?appliance|boxoban|babyai|minigrid|rtfm|scienceworld|hanabi|leduc|codenames", re.I)
NC_LIC = re.compile(r"\bnc\b|non.?commercial|research|cc.by.nc", re.I)


def norm_txt(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def shingles(s, n=10, every=1):
    w = norm_txt(s).split()
    if len(w) < n:
        return [hash(" ".join(w))] if len(w) >= 6 else []
    return [hash(" ".join(w[i:i + n])) for i in range(0, len(w) - n + 1, every)]


DI_TESTS = [  # (hf id, config, split, fields) -- test/eval splits of Decision Index public benchmarks we can fetch
    ("legacy-datasets/banking77", None, "test", ["text"]),
    ("clinc/clinc_oos", "plus", "test", ["text"]),
    ("allenai/ai2_arc", "ARC-Easy", "test", ["question"]),
    ("allenai/ai2_arc", "ARC-Challenge", "test", ["question"]),
    ("Rowan/hellaswag", None, "validation", ["ctx"]),
    ("allenai/winogrande", "winogrande_xl", "validation", ["sentence"]),
    ("allenai/openbookqa", "main", "test", ["question_stem"]),
    ("tau/commonsense_qa", None, "validation", ["question"]),
    ("facebook/anli", None, "test_r1", ["premise"]), ("facebook/anli", None, "test_r2", ["premise"]),
    ("facebook/anli", None, "test_r3", ["premise"]),
    ("cais/mmlu", "all", "test", ["question"]),
    ("TIGER-Lab/MMLU-Pro", None, "test", ["question"]),
    ("openai/gsm8k", "main", "test", ["question"]),
]


def di_keys(a):
    """10-word shingles of the eval items of the Decision Index public benchmarks (fetched from the Hub) and of
    OpenDecisions. Training rows sharing any shingle with them are dropped."""
    from datasets import load_dataset
    keys, report = set(), {}
    for rid, conf, split, fields in DI_TESTS:
        try:
            ds = load_dataset(rid, conf, split=split)
            n0 = len(keys)
            for r in ds:
                for f in fields:
                    keys.update(shingles(r[f], every=3))
            report[f"{rid}:{conf}:{split}"] = dict(items=len(ds), keys=len(keys) - n0)
        except Exception as e:
            report[f"{rid}:{conf}:{split}"] = f"FAIL {repr(e)[:150]}"
    od = [json.loads(l) for l in open(glob.glob(f"{a.raw}/text/opendecisions/*.jsonl")[0])]
    n0 = len(keys)
    for r in od:
        st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
        keys.update(shingles(st, every=2))
    report["opendecisions"] = dict(items=len(od), keys=len(keys) - n0)
    print(json.dumps(report, indent=1), flush=True)
    return keys, report


def _q(kind, instr, options, target, criteria_yes=None, criteria_no=None):
    if kind == "noul":
        p = float(target[0] if isinstance(target, (list, tuple)) else target)
        ins = instr.strip()
        if criteria_yes or criteria_no:
            ins += f"\n(yes = {criteria_yes or 'true'}; no = {criteria_no or 'false'})"
        return dict(type="noul", instr=ins, target=round(p, 4))
    return dict(type=kind, instr=instr.strip(), options=[str(o) for o in options], target=[round(float(x), 4) for x in target])


def _ok_q(q):
    if q["type"] == "noul":
        return 0.0 <= q["target"] <= 1.0
    n = len(q["options"])
    return 2 <= n <= 26 and len(q["target"]) == n and abs(sum(q["target"]) - 1) < 0.02


def load_tasksource(a, split, rng, max_rows):
    import pandas as pd
    fs = sorted(glob.glob(f"{a.raw}/text/tasksource/data/{split}-*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["state", "kind", "options", "target", "question", "source", "variant",
                                                 "split", "group_id", "license_use"]) for f in fs])
    st = dict(rows=len(df))
    df = df[df.license_use != "non-commercial"]; st["after_license"] = len(df)
    df = df[~df.source.str.contains(EXCL_SRC)]; st["after_source_excl"] = len(df)
    if split == "train":
        df = df[df.split == "train"]; st["after_split_train"] = len(df)
    groups = df.groupby(["group_id", "state"], sort=False)
    keys = list(groups.groups.keys())
    rng.shuffle(keys)
    recs, n = [], 0
    for k in keys:
        if n >= max_rows:
            break
        g = groups.get_group(k)
        qs = []
        for r in g.itertuples():
            q = _q(r.kind, r.question, list(r.options) if r.options is not None else [], list(r.target))
            if _ok_q(q):
                qs.append(q)
        if qs:
            recs.append(dict(src="tasksource", sub=g.source.iloc[0], text=k[1], questions=qs[:16]))
            n += len(qs[:16])
    st["sampled_questions"] = n
    return recs, st


def load_decisionbank(a, rng, max_rows):
    rows = []
    for f in sorted(glob.glob(f"{a.raw}/text/decisionbank/data/DecisionBank-260k/*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    st = dict(rows=len(rows))
    rows = [r for r in rows if not NC_LIC.search(str(r.get("source_license", "")))]; st["after_license"] = len(rows)
    rows = [r for r in rows if not EXCL_SRC.search(f"{r.get('source_dataset','')} {r.get('family','')}")]
    st["after_source_excl"] = len(rows)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["group_id"], r["state"])].append(r)
    keys = list(by); rng.shuffle(keys)
    recs, n, ncal, ncal_hard = [], 0, 0, 0
    cal, cal_hard = [], []
    for k in keys:
        qs = []
        for r in by[k]:
            opts = r["options"] or []
            tgt = r.get("target_distribution")
            if tgt is None:
                tgt = [1.0 if i == r["label"] else 0.0 for i in range(len(opts))]
            if r["qtype"] == "noul":
                ids = [str(o["id"]).lower() for o in opts]
                if "yes" in ids and "no" in ids and len(ids) == 2:
                    q = _q("noul", r["question"], None, [tgt[ids.index("yes")]],
                           opts[ids.index("yes")].get("description"), opts[ids.index("no")].get("description"))
                else:
                    q = _q("choice", r["question"], [o["description"] or o["id"] for o in opts], tgt)
            else:
                q = _q(r["qtype"], r["question"], [o["description"] or o["id"] for o in opts], tgt)
            if _ok_q(q):
                q["tier"] = r.get("tier")
                qs.append(q)
        if not qs:
            continue
        rec = dict(src="decisionbank", sub=by[k][0].get("part"), text=k[1], questions=qs[:16])
        if ncal_hard < a.calib_text_hard and all(q.get("tier") in ("judge", "hard") for q in rec["questions"]):
            cal_hard.append(rec); ncal_hard += len(rec["questions"])  # held-out, difficulty-matched temperature split
        elif ncal < a.calib_text_per_source:
            cal.append(rec); ncal += len(rec["questions"])
        elif n < max_rows:
            recs.append(rec); n += len(rec["questions"])
    st["sampled_questions"] = n
    st["calib_hard_questions"] = ncal_hard
    return recs, cal, cal_hard, st


def load_lemon(a, rng, max_rows):
    lic = [json.loads(l) for l in open(f"{a.raw}/text/lemon/row_licenses.jsonl")]
    st = dict(rows=len(lic))
    keep = []
    for i, l in enumerate(lic):
        if l.get("license_use") == "non-commercial":
            continue
        up = " ".join(str(v) for k, v in l.items() if k not in ("license_use", "notes"))
        if EXCL_SRC.search(up):
            continue
        keep.append(i)
    st["after_license_and_source_excl"] = len(keep)
    rng.shuffle(keep)
    want = set(keep[: int(max_rows * 1.3)])
    recs, n = [], 0
    with open(f"{a.raw}/text/lemon/train.jsonl") as f:
        for i, line in enumerate(f):
            if i not in want:
                continue
            r = json.loads(line)
            qq = r["question"]
            crit = qq.get("criteria") or {}
            kind = qq["type"]
            if kind == "noul":
                t = r["target"]
                try:
                    if isinstance(t, list) and len(t) == 2 and crit and len(crit) == 2:
                        q = _q("choice", qq["instructions"], list(crit.values()), t)
                    else:
                        q = _q("noul", qq["instructions"], None, t[-1] if isinstance(t, list) else t)
                except Exception:
                    continue
            else:
                ks = list(crit)  # keys "0","1",.. in upstream order (or named ids)
                t = r["target"]
                if isinstance(t, dict):
                    t = [t.get(k, 0.0) for k in ks]
                try:
                    q = _q(kind, qq["instructions"], [crit[k] for k in ks], t)
                except Exception:
                    continue
            if _ok_q(q):
                recs.append(dict(src="lemon", sub=lic[i].get("upstream_dataset"), text=r["state"], questions=[q]))
                n += 1
    rng.shuffle(recs)
    recs = recs[:max_rows]
    st["sampled_questions"] = len(recs)
    return recs, st


def text(a):
    from transformers import AutoTokenizer
    rng = random.Random(a.seed)
    tokz = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    keys, di_report = di_keys(a)
    stats = dict(di_keys=di_report)
    ts, stats["tasksource"] = load_tasksource(a, "train", rng, a.n_tasksource)
    ts_cal, _ = load_tasksource(a, "validation", rng, a.calib_text_per_source)
    db, db_cal, db_cal_hard, stats["decisionbank"] = load_decisionbank(a, rng, a.n_decisionbank)
    lm, stats["lemon"] = load_lemon(a, rng, a.n_lemon)
    pools = dict(train=ts + db + lm, calib=ts_cal + db_cal, calib_hard=db_cal_hard)
    seen = set()
    for name in ["calib", "calib_hard", "train"]:
        out, drop = [], collections.Counter()
        for r in pools[name]:
            if not isinstance(r["text"], str) or not r["text"].strip():
                drop["empty_state"] += 1; continue
            sk = shingles(r["text"], every=1)
            if any(h in keys for h in sk) or any(h in keys for q in r["questions"] for h in shingles(q["instr"], every=1)):
                drop["di_or_opendecisions_overlap"] += 1; continue
            qs = []
            for q in r["questions"]:
                k = hash((norm_txt(r["text"])[:2000], norm_txt(q["instr"]), tuple(norm_txt(o) for o in q.get("options", []))))
                if k in seen:
                    drop["dup_question"] += 1; continue
                seen.add(k); qs.append(q)
            if qs:
                out.append(dict(r, questions=qs))
        # token lengths (state, longest block) -> drop over-long
        enc = tokz([r["text"] for r in out], add_special_tokens=False)["input_ids"]
        res = []
        for r, ids in zip(out, enc):
            if len(ids) > a.max_state_tokens:
                drop["state_too_long"] += 1; continue
            qs = []
            for q in r["questions"]:
                body = q["instr"] + " ".join(q.get("options", []))
                nb = len(tokz(body, add_special_tokens=False)["input_ids"]) + 4 * len(q.get("options", [])) + 30
                if nb > a.max_q_tokens_typed:
                    drop["block_too_long"] += 1; continue
                qs.append(q)
            if qs:
                res.append(dict(r, plen=len(ids) + 4, questions=qs))
        write_jsonl(f"{a.out}/{dict(train='pool_text', calib='calib_text', calib_hard='calib_text_hard')[name]}.jsonl", res)
        stats[name] = dict(records=len(res), questions=sum(len(r["questions"]) for r in res), dropped=drop,
                           by_src=collections.Counter(r["src"] for r in res for _ in r["questions"]),
                           by_type=collections.Counter(q["type"] for r in res for q in r["questions"]))
    json.dump(stats, open(f"{a.out}/stats_text.json", "w"), indent=1, default=str)
    print(json.dumps(stats, indent=1, default=str)[:4000])


# ======================================================================================= GUI grounding
def _gui_items(a):
    """-> {domain: {abs_image_path: [ (instruction, box_norm xyxy | None, point_norm | None) ]}}"""
    G = f"{a.raw}/gui"
    files = {}
    for root in [f"{G}/osatlas/desktop_domain/img", f"{G}/osatlas/desktop_domain/img_win", f"{G}/osatlas/mobile_domain/img",
                 f"{G}/aguvis/img"]:
        for dp, _, fs in os.walk(root):
            for f in fs:
                files.setdefault((root, f), os.path.join(dp, f))
                files.setdefault((root, os.path.relpath(os.path.join(dp, f), root)), os.path.join(dp, f))
    def find(root, name):
        return files.get((root, name)) or files.get((root, os.path.basename(name)))
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    stats = collections.Counter()
    D, M, W_ = f"{G}/osatlas/desktop_domain/img", f"{G}/osatlas/mobile_domain/img", f"{G}/aguvis/img"
    DW = f"{G}/osatlas/desktop_domain/img_win"
    if os.path.exists(f"{G}/osatlas/desktop_domain/windows_splited.json"):
        for r in json.load(open(f"{G}/osatlas/desktop_domain/windows_splited.json")):
            p = find(DW, r["img_filename"])
            if not p:
                stats["missing_img_desktop_windows"] += 1; continue
            for e in r["elements"]:
                out["desktop_windows"][p].append((e["instruction"], e["bbox"], None))
    # GroundCUA (MIT): pixel boxes -> marked with a leading "px" flag, normalized in gui() once the size is known
    for jf in glob.glob(f"{G}/groundcua/data/*/*.json"):
        app, h = jf.split("/")[-2], os.path.splitext(os.path.basename(jf))[0]
        p = f"{G}/groundcua/images/{app}/{h}.png"
        if not os.path.exists(p):
            stats["missing_img_groundcua"] += 1; continue
        try:
            els = json.load(open(jf))
        except Exception:
            continue
        for e in els:
            t = (e.get("text") or "").strip()
            if t and e.get("bbox"):
                out["desktop_groundcua"][p].append((t, ("px", e["bbox"]), None))
    for dom, (jl, imgdir) in _jedi_sources(G).items():
        for p, t, b in _jedi_items(jl, imgdir):
            out[dom][p].append((t, b, None))
    for fn, dom in [("osatlas/desktop_domain/linux_splited.json", "desktop_linux"), ("osatlas/desktop_domain/macos_splited.json", "desktop_macos"),
                    ("osatlas/mobile_domain/uibert_raw.json", "mobile_uibert"), ("osatlas/mobile_domain/aw_mobile.json", "mobile_aw")]:
        root = D if "desktop" in fn else M
        for r in json.load(open(f"{G}/{fn}")):
            p = find(root, r["img_filename"])
            if not p:
                stats[f"missing_img_{dom}"] += 1; continue
            for e in r["elements"]:
                out[dom][p].append((e["instruction"], e["bbox"], None))
    for fn, dom in [("osatlas/mobile_domain/widget_captioning.json", "mobile_widgetcap"), ("osatlas/mobile_domain/ricosca.json", "mobile_ricosca")]:
        for r in json.load(open(f"{G}/{fn}")):
            p = find(M, r["img_filename"])
            if not p:
                stats[f"missing_img_{dom}"] += 1; continue
            out[dom][p].append((r["instruction"], r["bbox"], None))
    pat = re.compile(r"(?:click|doubleClick|rightClick|moveTo)\(x=([0-9.]+),\s*y=([0-9.]+)")
    for r in json.load(open(f"{G}/aguvis/guienv.json")):
        c = r["conversations"]
        ins = c[0]["value"].replace("<image>", "").strip()
        ins = re.sub(r"^(Double[- ]click on|Double[- ]click|Right[- ]click on|Right[- ]click|Click on|Click|Hover over|Move to|Move the mouse to|Hover on)\s+",
                     "", ins, flags=re.I).strip().strip("'\"")
        m = pat.search(c[1]["value"])
        p = find(W_, r["image"])
        if not m or not p:
            stats["guienv_skip"] += 1; continue
        out["web_guienv"][p].append((ins, None, (float(m.group(1)), float(m.group(2)))))
    stats.update({f"imgs_{k}": len(v) for k, v in out.items()})
    stats.update({f"elems_{k}": sum(len(x) for x in v.values()) for k, v in out.items()})
    return out, stats


def _jedi_sources(G):
    # Disabled in v1.1: Jedi's *_qwen25 files give absolute pixel coordinates in Qwen2.5-VL's smart-resized input frame
    # (click points / x,y,w,h), which we cannot map back to the stored screenshots reliably.
    return {}
    J = f"{G}/jedi"
    out = {}
    for dom, pat, img in [("desktop_jedi_layout", "datasets/layout/os_layout_v1_grounding/*.jsonl", "images/os_layout_data"),
                          ("desktop_jedi_component", "datasets/component/generated/component_v1_130k/*.jsonl", "images/component_v1_130k")]:
        fs = glob.glob(f"{J}/{pat}")
        if fs and os.path.isdir(f"{J}/{img}"):
            out[dom] = (fs[0], f"{J}/{img}")
    return out


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _jedi_items(jl, imgdir, max_items=400000):
    """Jedi conversation jsonl: {"image": rel, "conversations": [{"from": "human", "value": "<image>..instr"},
    {"from": "gpt", "value": "...<box or click with 0-1 coords>"}]}. Yields (path, instr, box_norm_or_None)."""
    files = {}
    for dp, _, fs in os.walk(imgdir):
        for f in fs:
            files.setdefault(f, os.path.join(dp, f))
    n = 0
    for line in open(jl):
        if n >= max_items:
            break
        try:
            r = json.loads(line)
            img = r.get("image") or r.get("images", [None])[0]
            p = files.get(os.path.basename(img))
            conv = r["conversations"]
        except Exception:
            continue
        if not p:
            continue
        for i in range(0, len(conv) - 1, 2):
            ins = re.sub(r"<image>", "", conv[i]["value"]).strip()
            nums = [float(x) for x in _NUM.findall(conv[i + 1]["value"])]
            if len(nums) == 4 and all(0 <= v <= 1 for v in nums) and nums[2] > nums[0] and nums[3] > nums[1]:
                yield p, ins, nums; n += 1


def gui(a):
    from PIL import Image
    rng = random.Random(a.seed)
    idx, _, _ = load_eval_index(a)
    items, st = _gui_items(a)
    st = dict(st)
    quota = dict(desktop_linux=a.gui_desktop // 2, desktop_macos=a.gui_desktop // 2, desktop_windows=a.gui_windows,
                 desktop_groundcua=a.gui_groundcua, desktop_jedi_layout=a.gui_jedi, desktop_jedi_component=a.gui_jedi,
                 mobile_aw=a.gui_mobile // 3,
                 mobile_uibert=a.gui_mobile // 6, mobile_widgetcap=a.gui_mobile // 4, mobile_ricosca=a.gui_mobile // 4,
                 web_guienv=a.gui_web)
    texts_by_dom = {d: [t for v in items[d].values() for t, _, _ in v if len(t.split()) >= 3] for d in items}
    recs, calib, dup, bad = [], [], 0, 0
    for dom, q in quota.items():
        imgs = list(items.get(dom, {}))
        rng.shuffle(imgs)
        n, n_img = 0, 0
        for p in imgs:
            if n >= q:
                break
            els = items[dom][p]
            try:
                h, (W, H) = phash_file(p)
            except Exception:
                bad += 1; continue
            if idx.hit(h):
                dup += 1; continue
            seen, qs = set(), []
            for t, b, pt in els:
                t = re.sub(r"\s+", " ", str(t)).strip()
                if not t or t.lower() in seen or len(t) > 200:
                    continue
                seen.add(t.lower())
                if b is not None:
                    if isinstance(b, tuple) and b[0] == "px":
                        b = [b[1][0] / W, b[1][1] / H, b[1][2] / W, b[1][3] / H]
                    x1, y1, x2, y2 = [min(max(v, 0.0), 1.0) for v in b]
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue
                    qs.append(dict(type="box", text=t, present=True, box=[x1 * W, y1 * H, x2 * W, y2 * H]))
                else:
                    qs.append(dict(type="box", text=t, present=True, box=None, points=[[pt[0] * W, pt[1] * H]]))
            if not qs:
                continue
            rng.shuffle(qs); qs = qs[: a.gui_max_q]
            # negatives (~gui_neg of questions): same-domain long phrases not overlapping any element text here (hard),
            # or phrases from another domain (easy)
            here = " | ".join(x[0].lower() for x in els)
            n_neg = max(0, round(len(qs) * a.gui_neg / (1 - a.gui_neg) + rng.random() - 0.5))
            for k in range(n_neg):
                hard = rng.random() < 0.5
                pool = texts_by_dom[dom] if hard else texts_by_dom[rng.choice([d for d in texts_by_dom if d.split("_")[0] != dom.split("_")[0]])]
                for _ in range(10):
                    t = rng.choice(pool)
                    tw = set(words(t))
                    if t.lower() in here or len(tw & set(words(here))) > len(tw) // 2:
                        continue
                    qs.append(dict(type="box", text=t, present=False, box=None, neg="hard" if hard else "easy"))
                    break
            rel = os.path.relpath(p, a.raw)
            r = dict(src="gui", sub=dom, img=rel, wh=[W, H], max_vis=a.gui_max_vis, plen=plen_image(W, H, a.gui_max_vis),
                     questions=qs)
            if n_img < a.gui_calib_per_dom:
                calib.append(r)
            else:
                recs.append(r); n += len(qs)
            n_img += 1
        st[f"used_{dom}"] = dict(images=n_img, questions=n)
    st["eval_hash_dup_removed"], st["unreadable"] = dup, bad
    rng.shuffle(recs)
    write_jsonl(f"{a.out}/pool_gui.jsonl", recs)
    write_jsonl(f"{a.out}/calib_gui.jsonl", calib)
    qs = [q for r in recs for q in r["questions"]]
    st.update(records=len(recs), questions=len(qs), absent=sum(not q["present"] for q in qs),
              hard_neg=sum(q.get("neg") == "hard" for q in qs), point_only=sum(q.get("box") is None and q["present"] for q in qs))
    json.dump(st, open(f"{a.out}/stats_gui.json", "w"), indent=1, default=str)
    print(json.dumps(st, indent=1, default=str))


# ======================================================================================= pointing / counting (PixMo)
COUNT_LEVELS = [str(i) for i in range(11)] + ["more than 10"]


def _count_q(label, n):
    k = min(int(n), 11)
    return dict(type="score", instr=f"How many {label} are in the image?", options=COUNT_LEVELS,
                target=[1.0 if i == k else 0.0 for i in range(len(COUNT_LEVELS))])


def _pts(v):
    """pixmo points cell -> [(x%, y%)]"""
    if v is None:
        return []
    if isinstance(v, dict):
        return list(zip([float(x) for x in v.get("x", [])], [float(y) for y in v.get("y", [])]))
    return [(float(p["x"]), float(p["y"])) for p in v]


def pixmo(a):
    import pandas as pd
    from PIL import Image
    rng = random.Random(a.seed)
    idx, ev_urls, ev_sha = load_eval_index(a)
    img_dir = f"{a.raw}/pixmo/img"
    os.makedirs(img_dir, exist_ok=True)
    pts = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.raw}/pixmo/points/data/train-*.parquet"))])
    cnt = pd.read_parquet(glob.glob(f"{a.raw}/pixmo/count/data/train-*.parquet")[0])
    st = dict(points_rows=len(pts), count_rows=len(cnt))
    urls_p = list(dict.fromkeys(pts.image_url)); rng.shuffle(urls_p)
    urls_c = list(dict.fromkeys(cnt.image_url)); rng.shuffle(urls_c)
    urls_p, urls_c = urls_p[: a.pixmo_point_images], urls_c[: a.pixmo_count_images]
    want = [u for u in dict.fromkeys(urls_p + urls_c) if u not in ev_urls]
    st["url_eval_overlap_removed"] = len(set(urls_p + urls_c) & ev_urls)
    fn = {u: f"{img_dir}/{hashlib.md5(u.encode()).hexdigest()}.jpg" for u in want}
    got = _download([(u, fn[u]) for u in want])
    ok = {u for u, g in zip(want, got) if g}
    st["download_ok"], st["download_tried"] = len(ok), len(want)
    oku = sorted(ok)
    hs = _hash_many([fn[u] for u in oku])
    good, wh = set(), {}
    n_dup = 0
    for u, h in zip(oku, hs):
        if h is None:
            continue
        if idx.hit(h):
            n_dup += 1; continue
        try:
            wh[u] = Image.open(fn[u]).size
        except Exception:
            continue
        good.add(u)
    st["eval_hash_dup_removed"] = n_dup
    recs = []
    maxv = a.pixmo_max_vis
    sub_p = pts[pts.image_url.isin(good) & ~pts.image_sha256.isin(ev_sha)]
    for u, g in sub_p.groupby("image_url"):
        W, H = wh[u]
        qs, seen = [], set()
        for r in g.itertuples():
            lab = str(r.label).strip()
            if not lab or lab.lower() in seen:
                continue
            seen.add(lab.lower())
            p = _pts(r.points)
            if p:
                qs.append(dict(type="box", text=lab, present=True, box=None,
                               points=[[x / 100 * W, y / 100 * H] for x, y in p[:64]]))
            else:
                qs.append(dict(type="box", text=lab, present=False, box=None, neg="natural"))
            if r.collection_method == "counting":
                c = r.count if r.count == r.count else len(p)
                try:
                    qs.append(_count_q(lab, float(c)))
                except Exception:
                    pass
        if qs:
            recs.append(dict(src="pixmo_points", img=os.path.relpath(fn[u], a.raw), wh=[W, H], max_vis=maxv,
                             plen=plen_image(W, H, maxv), questions=qs[:48]))
    sub_c = cnt[cnt.image_url.isin(good) & ~cnt.image_sha256.isin(ev_sha)]
    for u, g in sub_c.groupby("image_url"):
        W, H = wh[u]
        qs = []
        for r in g.itertuples():
            lab = str(r.label).strip()
            qs.append(_count_q(lab, float(r.count)))
            qs.append(dict(type="noul", instr=f"Is there at least one {lab} in the image?", target=float(float(r.count) > 0)))
        recs.append(dict(src="pixmo_count", img=os.path.relpath(fn[u], a.raw), wh=[W, H], max_vis=maxv,
                         plen=plen_image(W, H, maxv), questions=qs))
    rng.shuffle(recs)
    write_jsonl(f"{a.out}/pool_pixmo.jsonl", recs)
    qs = [q for r in recs for q in r["questions"]]
    st.update(records=len(recs), questions=len(qs), by_type=collections.Counter(q["type"] for q in qs),
              absent=sum(q["type"] == "box" and not q["present"] for q in qs))
    json.dump(st, open(f"{a.out}/stats_pixmo.json", "w"), indent=1, default=str)
    print(json.dumps(st, indent=1, default=str))


def postdedup(a):
    """Point-Bench (PointArena) images: unzip + hash, add to eval_hashes.json, and drop pixmo/gui pool records that
    match them (dHash) or whose source URL is in PointArena's pixmo_metadata.csv."""
    import csv, zipfile
    P = f"{a.raw}/evalimg/pointarena"
    if not os.path.isdir(f"{P}/img"):
        zipfile.ZipFile(f"{P}/selected_images.zip").extractall(f"{P}/img")
    pa = [p for p in glob.glob(f"{P}/img/**/*", recursive=True) if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    hs = [h for h in _hash_many(pa) if h]
    d = json.load(open(f"{a.out}/eval_hashes.json"))
    d["point_bench"] = hs
    json.dump(d, open(f"{a.out}/eval_hashes.json", "w"))
    pb = HashIndex(hs)
    urls = [r["image_url"] for r in csv.DictReader(open(f"{P}/pixmo_metadata.csv"))]
    bad_names = {hashlib.md5(u.encode()).hexdigest() for u in urls}
    st = dict(point_bench_images=len(pa), point_bench_hashes=len(hs))
    for pool in ["pool_pixmo", "pool_gui"]:
        fn = f"{a.out}/{pool}.jsonl"
        if not os.path.exists(fn):
            continue
        recs = [json.loads(l) for l in open(fn)]
        hh = _hash_many([f"{a.raw}/{r['img']}" for r in recs])
        keep = [r for r, h in zip(recs, hh) if h and not pb.hit(h)
                and os.path.splitext(os.path.basename(r["img"]))[0] not in bad_names]
        st[pool] = dict(before=len(recs), after=len(keep))
        write_jsonl(fn, keep)
    json.dump(st, open(f"{a.out}/stats_postdedup.json", "w"), indent=1)
    print(st)


# ======================================================================================= final mixture
def mix(a):
    rng = random.Random(a.seed)
    frac = dict(text=a.f_text, gui=a.f_gui, natural=a.f_natural, pixmo=a.f_pixmo)
    tot = a.total
    out, st = [], {}
    for name, f in frac.items():
        fn = f"{a.out}/pool_{name}.jsonl"
        if not os.path.exists(fn):
            print("missing pool", fn); continue
        recs = [json.loads(l) for l in open(fn)]
        rng.shuffle(recs)
        if name == "gui":  # per-domain resolution / image caps (questions per screenshot differ a lot by domain)
            vis = dict(kv.split("=") for kv in a.gui_vis.split(",")) if a.gui_vis else {}
            cap = dict(kv.split("=") for kv in a.gui_img_cap.split(",")) if a.gui_img_cap else {}
            seen, keep = collections.Counter(), []
            for r in recs:
                dom = r["sub"].split("_")[0]
                ck = r["sub"] if r["sub"] in cap else dom
                if ck in cap and seen[ck] >= int(cap[ck]):
                    continue
                seen[ck] += 1
                up = dict(kv.split("=") for kv in a.gui_upscale.split(",")) if a.gui_upscale else {}
                if dom in vis or dom in up:
                    r["max_vis"] = int(vis.get(dom, r["max_vis"]))
                    if dom in up:
                        r["upscale"] = float(up[dom])
                    r["plen"] = plen_image(*r["wh"], r["max_vis"], r.get("upscale", 2.0))
                keep.append(r)
            recs = keep
        want, n, take, rep = int(tot * f), 0, [], 0
        while n < want and rep < a.max_repeat:   # small pools (GUI) are repeated up to max_repeat times
            for r in recs:
                if n >= want:
                    break
                take.append(r); n += len(r["questions"])
            rep += 1
        out += take
        qs = [q for r in take for q in r["questions"]]
        st[name] = dict(records=len(take), questions=n, pool_questions=sum(len(r["questions"]) for r in recs), repeats=rep,
                        by_src=collections.Counter(r.get("sub") or r["src"] for r in take for _ in r["questions"]).most_common(40),
                        by_type=collections.Counter(q.get("type", "box") for q in qs),
                        absent=sum(q.get("type") == "box" and not q["present"] for q in qs),
                        hard_neg=sum(q.get("neg") == "hard" for q in qs), easy_neg=sum(q.get("neg") == "easy" for q in qs),
                        prefix_tokens=sum(r.get("plen", 0) for r in take),
                        block_tokens_est=sum(30 if q.get("type", "box") == "box" else
                                             (len(q["instr"]) + sum(len(str(o)) + 4 for o in q.get("options", []))) // 3 + 30
                                             for q in qs))
    rng.shuffle(out)
    write_jsonl(f"{a.out}/train.jsonl", out)
    allq = [q for r in out for q in r["questions"]]
    nb = sum(q.get("type", "box") == "box" for q in allq)
    st["total"] = dict(records=len(out), questions=len(allq), box_questions=nb,
                       absent=sum(q.get("type", "box") == "box" and not q["present"] for q in allq),
                       by_type=collections.Counter(q.get("type", "box") for q in allq),
                       prefix_tokens=sum(r.get("plen", 0) for r in out),
                       block_tokens_est=sum(v.get("block_tokens_est", 0) for v in st.values()))
    st["total"]["absent_frac_of_all"] = st["total"]["absent"] / len(allq)
    st["total"]["absent_frac_of_box"] = st["total"]["absent"] / max(1, nb)
    json.dump(st, open(f"{a.out}/stats.json", "w"), indent=1, default=str)
    print(json.dumps(st, indent=1, default=str)[:6000])


# ======================================================================================= eval-image hashes (dedup)
class HashIndex:
    """dHash near-duplicate index (Hamming <= max_d via 4x16-bit pigeonhole buckets)."""

    def __init__(self, hashes, max_d=3):
        self.max_d = max_d
        self.h = [int(x, 16) for x in hashes]
        self.b = [collections.defaultdict(list) for _ in range(4)]
        for i, v in enumerate(self.h):
            for k in range(4):
                self.b[k][(v >> (16 * k)) & 0xFFFF].append(i)

    def hit(self, hx):
        v = int(hx, 16)
        for k in range(4):
            for i in self.b[k].get((v >> (16 * k)) & 0xFFFF, ()):
                if bin(v ^ self.h[i]).count("1") <= self.max_d:
                    return True
        return False


def _hash_many(paths, workers=32):
    from concurrent.futures import ThreadPoolExecutor
    def f(p):
        try:
            return phash_file(p)[0]
        except Exception:
            return None
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(f, paths))


def evalhash(a):
    """Hash every image of every eval set we report: RefCOCO family + gRefCOCO val/test (COCO ids, from data/prep),
    ScreenSpot-v2, ScreenSpot-Pro, VenusBench-GD (evals/gui_data.py output under /content/gui), PixMo-count val/test
    (downloaded from their URLs), Point-Bench (PointArena/pointarena-data)."""
    out = {}
    coco_ids = set()
    for f in glob.glob("data/prep/eval_*.jsonl"):
        for l in open(f):
            coco_ids.add(json.loads(l)["image_id"])
    paths = [f"{a.raw}/coco/train2014/COCO_train2014_{i:012d}.jpg" for i in sorted(coco_ids)]
    out["refcoco_family"] = [h for h in _hash_many(paths) if h]
    for name in ["ss2", "sspro", "venus"]:
        fn = f"/content/gui/{name}.jsonl"
        if not os.path.exists(fn):
            print("missing", fn); continue
        recs = [json.loads(l) for l in open(fn)]
        out[name] = [h for h in _hash_many([os.path.join(r["image_root"], r["file"]) for r in recs]) if h]
    import pandas as pd
    os.makedirs(f"{a.raw}/pixmo/count_eval_img", exist_ok=True)
    ev = []
    for sp in ["validation", "test"]:
        for f in glob.glob(f"{a.raw}/pixmo/count/data/{sp}-*.parquet"):
            ev.append(pd.read_parquet(f))
    if ev:
        df = pd.concat(ev)
        out["pixmo_count_eval_urls"] = sorted(set(df.image_url))
        if "image_sha256" in df:
            out["pixmo_count_eval_sha"] = sorted(set(df.image_sha256))
        got = _download([(u, f"{a.raw}/pixmo/count_eval_img/{hashlib.md5(u.encode()).hexdigest()}.jpg") for u in df.image_url])
        out["pixmo_count_eval"] = [h for h in _hash_many([p for p in got if p]) if h]
    pa = glob.glob(f"{a.raw}/evalimg/pointarena/**/*.*", recursive=True)
    pa = [p for p in pa if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    out["point_bench"] = [h for h in _hash_many(pa) if h]
    print({k: len(v) for k, v in out.items()})
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/eval_hashes.json", "w"))


def load_eval_index(a):
    d = json.load(open(f"{a.out}/eval_hashes.json"))
    hs = [h for k, v in d.items() if not k.endswith(("_urls", "_sha")) for h in v]
    return HashIndex(hs), set(d.get("pixmo_count_eval_urls", [])), set(d.get("pixmo_count_eval_sha", []))


def _download(jobs, workers=64, timeout=15):
    """jobs: [(url, dest)] -> list of dest or None. Skips existing files."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    def f(j):
        u, d = j
        if os.path.exists(d) and os.path.getsize(d) > 0:
            return d
        from PIL import Image
        hdr = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
               "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"}
        for attempt, url in enumerate([u, u, u.replace("_h.jpg", "_b.jpg") if "staticflickr" in u else u]):
            try:
                try:
                    import requests
                    rr = requests.get(url, headers=hdr, timeout=timeout * (1 + attempt))
                    rr.raise_for_status(); data = rr.content
                except ImportError:
                    data = urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout).read()
                im = Image.open(io.BytesIO(data)); im.verify()
                with open(d, "wb") as fh:
                    fh.write(data)
                return d
            except Exception:
                continue
        return None
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(f, jobs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--ann", default="data/ann")
    ap.add_argument("--raw", default="/content/data")
    ap.add_argument("--out", default="data/prep_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--absent_ratio", type=float, default=0.28)
    ap.add_argument("--hard_frac", type=float, default=0.65)
    ap.add_argument("--calib_images", type=int, default=600)
    ap.add_argument("--n_tasksource", type=int, default=220000)
    ap.add_argument("--n_decisionbank", type=int, default=200000)
    ap.add_argument("--n_lemon", type=int, default=120000)
    ap.add_argument("--calib_text_per_source", type=int, default=3000)
    ap.add_argument("--calib_text_hard", type=int, default=3000)
    ap.add_argument("--max_state_tokens", type=int, default=1536)
    ap.add_argument("--total", type=int, default=1000000)
    ap.add_argument("--gui_vis", default="desktop=6400,mobile=1280,web=1280")
    ap.add_argument("--max_repeat", type=int, default=2)
    ap.add_argument("--gui_img_cap", default="web=8000,mobile_uibert=1500")
    ap.add_argument("--gui_desktop", type=int, default=60000)
    ap.add_argument("--gui_windows", type=int, default=80000)
    ap.add_argument("--gui_groundcua", type=int, default=120000)
    ap.add_argument("--gui_jedi", type=int, default=20000)
    ap.add_argument("--gui_upscale", default="desktop=1.0,mobile=1.0,web=1.0")
    ap.add_argument("--gui_mobile", type=int, default=90000)
    ap.add_argument("--gui_web", type=int, default=90000)
    ap.add_argument("--gui_max_q", type=int, default=48)
    ap.add_argument("--gui_neg", type=float, default=0.10)
    ap.add_argument("--gui_max_vis", type=int, default=2048)
    ap.add_argument("--gui_calib_per_dom", type=int, default=60)
    ap.add_argument("--pixmo_point_images", type=int, default=30000)
    ap.add_argument("--pixmo_count_images", type=int, default=20000)
    ap.add_argument("--pixmo_max_vis", type=int, default=1024)
    ap.add_argument("--f_text", type=float, default=0.40)
    ap.add_argument("--f_gui", type=float, default=0.25)
    ap.add_argument("--f_natural", type=float, default=0.22)
    ap.add_argument("--f_pixmo", type=float, default=0.13)
    ap.add_argument("--max_q_tokens_typed", type=int, default=384)
    a = ap.parse_args()
    globals()[a.step](a)
