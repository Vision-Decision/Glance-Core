"""v1.3 training mixture = v1.2 mixture re-balanced per outputs/v1.2/regression_diagnosis.md.

  python scripts/v13/build_mix_v13.py --prep /content/data/prep_v1 --gd /content/gd --catfree12 /content/v12/out/catfree.jsonl \
      --neg13 /content/v13/out/catfree.jsonl --count /content/v13/count_pos.jsonl --zh /content/v13/zh_trans.jsonl \
      --pool_natural /content/data/prep_v1/pool_natural.jsonl --eval_ids /content/v12/eval_image_ids.json --out /content/data/prep_v13

Changes vs v1.2 (questions):
  RefCOCO/+/g human replay      79k -> ~220k   (base natural share; v1.1 level)
  Chinese RefCOCO translations  0 -> ~60k, each in a consistency group with its English original (anchor)
  Glance-Data zh hard negatives ~45k -> ~30k  (zh hneg keep-rate x0.6)
  relation negatives            16k -> = relation positives (v1.2 kinds + absent-class + "thing" negatives)
  generic pos : neg             54k:16k -> ~40k:27k
  pos_pos_sole                  32k -> 24k, paired with neg_position 12k + "position word + absent category" 12k
  count minimal pairs           0 -> ~15k present "two clocks" on images with exactly two clocks
  multi-instance bare names     0 -> ~20k (Glance-Data `full` cat_all; presence-only for the single-box head)
Same decontamination assert as v1.2.
"""
import argparse, collections, glob, json, os, random, subprocess, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

GD_KEEP = dict(short=0.09, medium=0.09, long=0.09, casual=0.09, cat_only=0.65, function=0.5, pos_pos_sole=0.38,
               neg_position=0.19, neg_count=0.25, hneg_attribute=0.12, hneg_relation=0.15, neg_near=0.12)
ZH_HNEG_SCALE = 0.6
ORIG_PRI = [("medium", "en"), ("short", "en"), ("long", "en"), ("casual", "en")]


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
    for k in ("prep", "gd", "catfree12", "neg13", "count", "zh", "pool_natural", "eval_ids", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--base_total", type=int, default=790000)
    ap.add_argument("--base_frac", default="text=0.38,gui=0.24,natural=0.279,pixmo=0.101")
    ap.add_argument("--generic_pos", type=int, default=40000); ap.add_argument("--generic_neg", type=int, default=27000)
    ap.add_argument("--pos_absent", type=int, default=12000); ap.add_argument("--count_q", type=int, default=15000)
    ap.add_argument("--cat_all", type=int, default=20000); ap.add_argument("--zh_pairs", type=int, default=60000)
    ap.add_argument("--max_q", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    st = {}
    # 1. base replay (v1.1 pools; natural = RefCOCO/+/g human expressions + their v1 negatives, back to v1.1 level)
    if not os.path.exists(f"{a.out}/train_base.jsonl"):
        fr = dict(kv.split("=") for kv in a.base_frac.split(","))
        cmd = [sys.executable, "-m", "dv.data.prep_v1", "mix", "--out", a.prep, "--total", str(a.base_total)] + \
              sum([[f"--f_{k}", v] for k, v in fr.items()], [])
        subprocess.run(cmd, check=True, stdout=open(f"{a.out}/log_base_mix.txt", "w"), stderr=subprocess.STDOUT)
        os.replace(f"{a.prep}/train.jsonl", f"{a.out}/train_base.jsonl"); os.replace(f"{a.prep}/stats.json", f"{a.out}/stats_base.json")
    recs = [json.loads(l) for l in open(f"{a.out}/train_base.jsonl")]
    # 2. Glance-Data single view (zh hard negatives scaled down) + `full` cat_all (multi-instance bare names)
    by_img, meta = collections.defaultdict(list), {}
    for fn in sorted(glob.glob(f"{a.gd}/single/*.jsonl")):
        for l in open(fn):
            r = json.loads(l)
            meta[r["image_id"]] = {k: r[k] for k in ("src", "img", "wh", "max_vis", "plen", "image_id")}
            for q in r["questions"]:
                p = GD_KEEP.get(q["qtype"], 0.0) * (ZH_HNEG_SCALE if q["qtype"].startswith("hneg") and q["lang"] == "zh" else 1.0)
                if rng.random() < p:
                    by_img[r["image_id"]].append(q)
    cat_all = []
    for fn in sorted(glob.glob(f"{a.gd}/full/*.jsonl")):
        for l in open(fn):
            r = json.loads(l)
            meta.setdefault(r["image_id"], {k: r[k] for k in ("src", "img", "wh", "max_vis", "plen", "image_id")})
            cat_all += [(r["image_id"], q) for q in r["questions"] if q["qtype"] == "cat_all" and q["n_targets"] >= 2]
    rng.shuffle(cat_all)
    for iid, q in cat_all[: a.cat_all]:
        by_img[iid].append(q)
    cons = collections.Counter()
    for iid, qs in by_img.items():
        tgt = collections.defaultdict(list)
        for q in qs:
            if q["present"] and q.get("box") is not None:
                tgt[tuple(round(v) for v in q["box"])].append(q)
        for g in tgt.values():
            orig = next((q for qt, lg in ORIG_PRI for q in g if q["qtype"] == qt and q["lang"] == lg), None)
            if len(g) < 2 or orig is None:
                continue
            for q in g:
                q["grp"] = f"gd{iid}:{id(orig)}"
            orig["anchor"] = True
            cons["groups"] += 1
        recs += pack(meta[iid], qs, a.max_q, rng)
    st["glance_data"] = dict(images=len(by_img), consistency=dict(cons), cat_all=min(a.cat_all, len(cat_all)))
    # 3. category-free (v1.2) re-balanced + v1.3 negatives
    cf12 = [json.loads(l) for l in open(a.catfree12)]
    neg13 = [json.loads(l) for l in open(a.neg13)]
    gp = [(i, j) for i, r in enumerate(cf12) for j, q in enumerate(r["questions"]) if q["qtype"] == "cf_generic"]
    rng.shuffle(gp)
    drop = set(gp[a.generic_pos:])  # refined below once the negative counts are known
    n_rel_pos = sum(q["qtype"] == "cf_rel" for r in cf12 for q in r["questions"])
    n_rel_neg12 = sum(q["qtype"] == "cf_rel_neg" for r in cf12 for q in r["questions"])
    n_gen_neg12 = sum(q["qtype"] == "cf_generic_neg" for r in cf12 for q in r["questions"])
    want = {"cf_rel_neg_absent": None, "cf_rel_neg_thing": None, "cf_generic_neg": max(0, a.generic_neg - n_gen_neg12),
            "cf_pos_absent": a.pos_absent}
    rel_need = max(0, n_rel_pos - n_rel_neg12)
    pool13 = collections.defaultdict(list)
    for i, r in enumerate(neg13):
        for j, q in enumerate(r["questions"]):
            pool13[q["qtype"]].append((i, j))
    for k in pool13:
        rng.shuffle(pool13[k])
    na, nt = len(pool13["cf_rel_neg_absent"]), len(pool13["cf_rel_neg_thing"])
    want["cf_rel_neg_absent"] = min(na, max(rel_need - min(nt, rel_need // 2), rel_need // 2))
    want["cf_rel_neg_thing"] = min(nt, rel_need - want["cf_rel_neg_absent"])
    keep13 = {x for k, n in want.items() for x in pool13.get(k, [])[:n]}
    # if the v1.3 negatives fall short of the targets, hold the ratios by dropping positives instead:
    # relation pos:neg 1:1, generic pos:neg 1.5:1
    rel_neg_tot = n_rel_neg12 + want["cf_rel_neg_absent"] + want["cf_rel_neg_thing"]
    gen_neg_tot = n_gen_neg12 + min(want["cf_generic_neg"], len(pool13.get("cf_generic_neg", [])))
    rp = [(i, j) for i, r in enumerate(cf12) for j, q in enumerate(r["questions"]) if q["qtype"] == "cf_rel"]
    rng.shuffle(rp)
    drop |= set(rp[rel_neg_tot:])
    drop |= set(gp[min(a.generic_pos, int(1.5 * gen_neg_tot)):])
    st["ratio_hold"] = dict(rel_pos_kept=min(len(rp), rel_neg_tot), rel_neg=rel_neg_tot,
                            generic_pos_kept=min(a.generic_pos, int(1.5 * gen_neg_tot), len(gp)), generic_neg=gen_neg_tot)
    for i, r in enumerate(cf12):
        qs = [q for j, q in enumerate(r["questions"]) if (i, j) not in drop]
        anchors_used = {q.get("grp") for q in qs if not q.get("anchor")}
        qs = [q for q in qs if not q.get("anchor") or q.get("grp") in anchors_used]
        if qs:
            recs.append(dict(r, questions=qs))
    for i, r in enumerate(neg13):
        qs = [q for j, q in enumerate(r["questions"]) if (i, j) in keep13]
        if qs:
            recs.append(dict(r, questions=qs))
    st["catfree"] = dict(generic_pos=min(a.generic_pos, len(gp)), rel_pos=n_rel_pos, v12_rel_neg=n_rel_neg12,
                         v12_generic_neg=n_gen_neg12, v13_added={k: min(n, len(pool13.get(k, []))) for k, n in want.items()})
    # 4. count minimal pairs
    cnt = [json.loads(l) for l in open(a.count)]
    rng.shuffle(cnt)
    nq, cr = 0, []
    for r in cnt:
        if nq >= a.count_q:
            break
        cr.append(r); nq += len(r["questions"])
    recs += cr
    st["count_pos"] = nq
    # 5. Chinese translations of RefCOCO train expressions, grouped with their English originals (anchor = English)
    pool = {}
    for l in open(a.pool_natural):
        r = json.loads(l)
        pool.setdefault(r["image_id"], dict(r, questions=[]))["questions"] += r["questions"]
    zh, seen_k = [], set()
    for fn in a.zh.split(","):  # priority order: Claude Opus 5.5 translations first, Qwen3.8-27B for the rest
        if not os.path.exists(fn):
            continue
        for l in open(fn):
            z = json.loads(l)
            if z["key"] not in seen_k:
                seen_k.add(z["key"]); zh.append(z)
    rng.shuffle(zh)
    st["zh_by_model"] = dict(collections.Counter(z.get("model", "qwen3.8-27b-fp8") for z in zh[: a.zh_pairs]))
    by = collections.defaultdict(list)
    for z in zh[: a.zh_pairs]:
        en = next((q for q in pool.get(z["image_id"], {}).get("questions", []) if q["text"] == z["en"] and q.get("box")), None)
        if en is None:
            continue
        g = f"zh{z['image_id']}:{len(by[z['image_id']])}"
        by[z["image_id"]] += [dict(en, grp=g, anchor=True, lang="en", qtype="ref_en"),
                              dict(en, text=z["zh"], grp=g, lang="zh", qtype="ref_zh", src=en.get("src", "") + "_zh")]
    for iid, qs in by.items():
        base = {k: pool[iid][k] for k in ("src", "img", "wh", "max_vis", "plen", "image_id")}
        recs += pack(dict(base, src="ref_zh"), qs, a.max_q, rng)
    st["zh_pairs"] = sum(len(v) for v in by.values()) // 2
    # 6. decontamination + write
    ev = json.load(open(a.eval_ids))
    ids = {r["image_id"] for r in recs if r.get("image_id") is not None and str(r.get("img", "")).startswith("coco/")}
    st["decontam_overlap"] = {k: len(ids & set(v)) for k, v in ev.items()}
    assert not any(st["decontam_overlap"].values()), st["decontam_overlap"]
    rng.shuffle(recs)
    with open(f"{a.out}/train.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    qt = collections.Counter()
    for r in recs:
        for q in r["questions"]:
            qt[q.get("qtype") or q.get("src") or q.get("type", "box")] += 1
    st["total"] = dict(records=len(recs), questions=sum(qt.values()))
    st["by_qtype"] = dict(qt.most_common())
    json.dump(st, open(f"{a.out}/stats_v13.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(st, indent=1, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main()
