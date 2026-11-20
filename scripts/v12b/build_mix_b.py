"""v1.2b (multi-box on top of v1.2) training mixture.

  python scripts/v12b/build_mix_b.py --pilot /content/v12b/train_mb.jsonl --gd_full /content/gd/full \
      --v12 /content/data/prep_v12/train.jsonl --eval_ids /content/v12/eval_image_ids.json --out /content/data/prep_v12b

Parts (questions):
  pilot      scripts/multibox/prep_pilot.py output: gRefCOCO multi-target train expressions, COCO "category = all
             instances" (+ absent categories), RefCOCO single-target replay  (all of it)
  gd_full    Glance-Data `full` view, qtype keep-rates --gd_keep: the multi-target kinds (group, cat_all, gref_multi,
             multi-target function) plus the count / position negatives and sole-object positives the pilot lacked
             (its neg_count N-acc collapsed 48.6 -> 5 without count negatives), and short phrasings (single-word
             expressions are where the pilot lost single-target accuracy)
  replay     a share of the v1.2 mixture: category-free records (with their consistency groups), v1 natural pool,
             text decisions and GUI (the joint stage still moves the LoRA through the single-box losses)
Decontamination: the same eval-image assert as scripts/v12/build_mix.py.
"""
import argparse, collections, glob, json, os, random

GD_KEEP = dict(group=0.5, cat_all=0.5, gref_multi=0.3, function=0.5, neg_count=0.3, pos_pos_sole=0.25, neg_position=0.25,
               short=0.12, cat_only=0.4, medium=0.04, long=0.03, casual=0.03, hneg_attribute=0.06, hneg_relation=0.06,
               neg_near=0.06)
REPLAY = dict(catfree=0.3, natural=0.5, glance_data=0.0, text=0.08, gui=0.10, pixmo=0.05)


def part(r):
    s = r.get("src")
    if s == "catfree":
        return "catfree"
    if s in ("glance_data", "gap_rule"):
        return "glance_data"
    if s == "natural":
        return "natural"
    if not r.get("img"):
        return "text"
    return "pixmo" if "pixmo" in r["img"] else "gui"


def pack(base, qs, maxq, rng):
    groups = collections.OrderedDict()
    for q in qs:
        groups.setdefault(q.get("grp") or id(q), []).append(q)
    gl = list(groups.values()); rng.shuffle(gl)
    out, cur = [], []
    for g in gl:
        if cur and len(cur) + len(g) > maxq:
            out.append(dict(base, questions=cur)); cur = []
        cur += g[:maxq]
    if cur:
        out.append(dict(base, questions=cur))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True); ap.add_argument("--gd_full", required=True); ap.add_argument("--v12", required=True)
    ap.add_argument("--eval_ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max_q", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    recs, st = [], {}
    pilot = [json.loads(l) for l in open(a.pilot)]
    recs += pilot
    st["pilot"] = dict(records=len(pilot), questions=sum(len(r["questions"]) for r in pilot))
    by_img, meta = collections.defaultdict(list), {}
    for fn in sorted(glob.glob(f"{a.gd_full}/*.jsonl")):
        for l in open(fn):
            r = json.loads(l)
            meta[r["image_id"]] = {k: r[k] for k in ("src", "img", "wh", "max_vis", "plen", "image_id")}
            for q in r["questions"]:
                if rng.random() < GD_KEEP.get(q["qtype"], 0.0):
                    by_img[r["image_id"]].append(q)
    gd = [x for iid, qs in by_img.items() for x in pack(meta[iid], qs, a.max_q, rng)]
    recs += gd
    qs = [q for r in gd for q in r["questions"]]
    st["gd_full"] = dict(records=len(gd), questions=len(qs), qtype=dict(collections.Counter(q["qtype"] for q in qs)),
                         n_targets=dict(collections.Counter(min(q["n_targets"], 4) for q in qs)))
    rep = collections.Counter()
    for l in open(a.v12):
        r = json.loads(l)
        p = part(r)
        if rng.random() < REPLAY.get(p, 0.0):
            recs.append(r); rep[p] += len(r["questions"])
    st["replay_questions"] = dict(rep)
    ev = json.load(open(a.eval_ids))
    ids = {r["image_id"] for r in recs if r.get("image_id") is not None and str(r.get("img", "")).startswith("coco/")}
    st["decontam_overlap"] = {k: len(ids & set(v)) for k, v in ev.items()}
    assert not any(st["decontam_overlap"].values()), st["decontam_overlap"]
    rng.shuffle(recs)
    with open(f"{a.out}/train.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st["total"] = dict(records=len(recs), questions=sum(len(r["questions"]) for r in recs))
    json.dump(st, open(f"{a.out}/stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(st, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
