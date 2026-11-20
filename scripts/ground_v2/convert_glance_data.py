"""Vision-Decision/Glance-Data -> Glance training / eval records (dv/data/mixed.py box format + multi-box fields),
plus the one gap Glance-Data lacks, filled by rules: position words / superlatives on the SOLE instance of a category
as positives ("the leftmost cat" with one cat; v1 rejects 86.5% of these), balanced against Glance-Data's neg_position.

  python scripts/ground_v2/convert_glance_data.py --gd ~/gv2/glance_data --ann ~/gv2/ann --out ~/gv2/gd_conv

Outputs ({out}/):
  full/train-0000K.jsonl    every question; records of <= 48 questions (multi-box head)
  single/train-0000K.jsonl  only n_targets <= 1 (current single-box head; multi-target questions dropped)
  diagnostic.jsonl          the 300-image diagnostic split, one record per image with all its questions (eval)
  stats.json

Question = {"type": "box", "text", "present", "box" (single target only, else null), "boxes", "n_targets",
            "qtype" (= Glance-Data `kind`, or pos_pos_sole), "lang", "neg", "src"}

Gap filler (qtype pos_pos_sole, src gap_rule), per image, from COCO train2014 instances:
  a category qualifies when it has exactly one instance, that instance is not crowd / tiny / a border fragment
  (area >= 1%, not (touching the border and < 3%)), and no other COCO instance of it exists (crowd included);
  positives: "the X on the left/right" + "左边/右边的X" only when the box centre is clearly on that side (< 0.4 / > 0.6)
  and the box does not cross the middle; superlatives that hold for any sole instance ("the leftmost X", "the biggest X",
  "the only X", "最左边的X", "最大的X", "唯一的那只X" ...). Sampled so that pos_pos_sole ~= --balance x neg_position
  overall: the model must learn the geometry, not "position word => absent" nor its opposite.
"""
import argparse, collections, glob, json, os, random, sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lexicon import EN, ZH, ZH_MW  # noqa: E402
from dv.data.prep_v1 import plen_image  # noqa: E402

NEG_KINDS = {"hneg_attribute", "hneg_relation", "neg_count", "neg_position", "neg_near"}


def q_of(q):
    boxes = [[round(float(v), 2) for v in b] for b in q["boxes"]]
    present = bool(q["present"])
    n = int(q["n_targets"])
    assert n == len(boxes) and (n > 0) == present, q
    return dict(type="box", text=str(q["text"]), present=present, box=boxes[0] if n == 1 else None, boxes=boxes,
                n_targets=n, qtype=str(q["kind"]), lang=str(q["lang"]), neg=("hard" if q["kind"] in NEG_KINDS else None),
                src=str(q["src"]))


def sole_position_questions(objs, W, H, rng):
    """objs: COCO annotations of one image -> candidate pos_pos_sole questions (en + zh)"""
    by = collections.defaultdict(list)
    for o in objs:
        by[o["category_id"]].append(o)
    out = []
    for cid, l in by.items():
        if len(l) != 1 or l[0]["iscrowd"]:
            continue
        x, y, w, h = l[0]["bbox"]
        area = w * h / (W * H)
        edge = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        if area < 0.01 or (edge and area < 0.03):
            continue
        c = CAT[cid]; en, zh, mw = EN[c], ZH[c], ZH_MW[c]
        box = [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]
        cx = (x + w / 2) / W
        en_opts = [f"the leftmost {en}", f"the rightmost {en}", f"the biggest {en}", f"the largest {en}", f"the smallest {en}",
                   f"the only {en}", f"the {en} closest to the camera", f"the {en} furthest to the left", f"the {en} furthest to the right"]
        zh_opts = [f"最左边的{zh}", f"最右边的{zh}", f"最大的{zh}", f"最小的{zh}", f"唯一的那{mw}{zh}", f"离镜头最近的{zh}"]
        side = []
        if cx < 0.4 and x + w < 0.55 * W:
            side = [("en", f"the {en} on the left"), ("en", f"the {en} on the left side"), ("zh", f"左边的{zh}"), ("zh", f"左边那{mw}{zh}")]
        elif cx > 0.6 and x > 0.45 * W:
            side = [("en", f"the {en} on the right"), ("en", f"the {en} on the right side"), ("zh", f"右边的{zh}"), ("zh", f"右边那{mw}{zh}")]
        cands = [("en", rng.choice(en_opts)), ("zh", rng.choice(zh_opts))]
        if side:  # a correct side word is the direct counterpart of Glance-Data's neg_position
            cands += [rng.choice([s for s in side if s[0] == "en"]), rng.choice([s for s in side if s[0] == "zh"])]
        for lang, t in cands:
            out.append(dict(type="box", text=t, present=True, box=box, boxes=[box], n_targets=1, qtype="pos_pos_sole",
                            lang=lang, neg=None, src="gap_rule"))
    return out


def load(gd, split):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{gd}/data/{split}-*.parquet"))])


def main():
    global CAT
    ap = argparse.ArgumentParser()
    ap.add_argument("--gd", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--balance", type=float, default=1.0, help="pos_pos_sole / neg_position target ratio")
    ap.add_argument("--shard_images", type=int, default=5000)
    ap.add_argument("--max_q", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    coco = json.load(open(f"{a.ann}/annotations/instances_train2014.json"))
    CAT = {c["id"]: c["name"] for c in coco["categories"]}
    anns = collections.defaultdict(list)
    for an in coco["annotations"]:
        anns[an["image_id"]].append(an)
    st = collections.defaultdict(collections.Counter)

    # ---- diagnostic (eval): one record per image, every question
    os.makedirs(f"{a.out}/full", exist_ok=True); os.makedirs(f"{a.out}/single", exist_ok=True)
    with open(f"{a.out}/diagnostic.jsonl", "w") as f:
        for r in load(a.gd, "diagnostic").itertuples():
            qs = [q_of(q) for q in r.questions]
            for q in qs:
                st["diag_qtype"][f"{q['qtype']}/{q['lang']}"] += 1
            f.write(json.dumps(dict(src="glance_data", img=f"coco/train2014/{r.file}", wh=[int(r.width), int(r.height)], max_vis=640,
                                    plen=plen_image(int(r.width), int(r.height), 640), image_id=int(r.image_id), questions=qs),
                               ensure_ascii=False) + "\n")
    # ---- diagnostic add-on: every sole-instance position / superlative positive on the same 300 images (Glance-Data's
    # diagnostic split has neg_position but no matching positives); separate file so the original split stays comparable
    with open(f"{a.out}/diagnostic_pos_sole.jsonl", "w") as f:
        for r in load(a.gd, "diagnostic").itertuples():
            qs = sole_position_questions(anns[int(r.image_id)], int(r.width), int(r.height), random.Random(int(r.image_id)))
            st["diag_qtype"]["pos_pos_sole"] += len(qs)
            if qs:
                f.write(json.dumps(dict(src="gap_rule", img=f"coco/train2014/{r.file}", wh=[int(r.width), int(r.height)], max_vis=640,
                                        plen=plen_image(int(r.width), int(r.height), 640), image_id=int(r.image_id), questions=qs),
                                   ensure_ascii=False) + "\n")
    # ---- train
    df = load(a.gd, "train")
    rows = list(df.itertuples())
    n_negpos = sum(1 for r in rows for q in r.questions if q["kind"] == "neg_position")
    gap = {r.image_id: sole_position_questions(anns[int(r.image_id)], int(r.width), int(r.height), rng) for r in rows}
    n_cand = sum(len(v) for v in gap.values())
    keep_p = min(1.0, a.balance * n_negpos / max(1, n_cand))
    st["gap"].update(neg_position=n_negpos, pos_pos_sole_candidates=n_cand)
    shard, fs = -1, {}
    for i, r in enumerate(rows):
        if i % a.shard_images == 0:
            for h in fs.values():
                h.close()
            shard += 1
            fs = {v: open(f"{a.out}/{v}/train-{shard:05d}.jsonl", "w") for v in ("full", "single")}
        qs = [q_of(q) for q in r.questions]
        qs += [q for q in gap[r.image_id] if rng.random() < keep_p]
        texts = collections.defaultdict(set)
        for q in qs:
            texts[q["text"].lower()].add(q["present"])
        bad = {t for t, v in texts.items() if len(v) > 1}  # a gap positive that contradicts a Glance-Data negative
        if bad:
            st["gap"]["conflict_dropped"] += sum(q["text"].lower() in bad for q in qs)
            qs = [q for q in qs if q["text"].lower() not in bad]
        base = dict(src="glance_data", img=f"coco/train2014/{r.file}", wh=[int(r.width), int(r.height)], max_vis=640,
                    plen=plen_image(int(r.width), int(r.height), 640), image_id=int(r.image_id))
        for view, sel in (("full", qs), ("single", [q for q in qs if q["n_targets"] <= 1])):
            sel = list(sel); rng.shuffle(sel)
            for j in range(0, len(sel), a.max_q):
                fs[view].write(json.dumps(dict(base, questions=sel[j:j + a.max_q]), ensure_ascii=False) + "\n")
                st[f"{view}_records"]["n"] += 1
            for q in sel:
                st[f"{view}_qtype"][q["qtype"]] += 1
                st[f"{view}_lang"][q["lang"]] += 1
                st[f"{view}_present"][str(q["present"])] += 1
    for h in fs.values():
        h.close()
    st["images"]["train"] = len(rows)
    out = {k: dict(v) for k, v in st.items()}
    json.dump(out, open(f"{a.out}/stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
