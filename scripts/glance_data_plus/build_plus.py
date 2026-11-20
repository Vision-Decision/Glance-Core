"""Glance-Data-Plus: our train-side supplements to Vision-Decision/Glance-Data, in Glance-Data's schema plus a `source`
column, so the two concatenate.  Private; publishing is the user's decision.  NO evaluation data in here.

  python scripts/glance_data_plus/build_plus.py --gd_full /content/gd/full --catfree12 /content/v12/out/catfree.jsonl \
      --neg13 /content/v13/out/catfree.jsonl --count /content/v13/count_pos.jsonl --zh /content/v13/zh_trans.jsonl \
      --pool_natural /content/data/prep_v1/pool_natural.jsonl --replay /content/data/prep_v13/train_base.jsonl \
      --eval_ids /content/v12/eval_image_ids.json --commit <hash> --out /content/gdplus

Row = one image x one source: image_id, file, width, height, questions[{text, lang, kind, src, present, n_targets, boxes}],
source.  Chinese RefCOCO translations: one row per pair (the zh question only; the English original is a RefCOCO train
expression and is referenced, not redistributed, in manifest/refcoco_zh_pairs.jsonl: image_id, box, dataset, sha1 of the
English text).  The human-expression replay is a manifest only (manifest/refcoco_replay.jsonl).
"""
import argparse, collections, glob, hashlib, json, os

import pandas as pd

KIND = {"cf_func": "function_catfree", "cf_generic": "generic", "cf_rel": "relation_generic", "cf_anchor": "anchor_category",
        "cf_func_neg": "neg_function_absent", "cf_generic_neg": "neg_generic_absent", "cf_rel_neg": "neg_relation_generic",
        "cf_rel_neg_absent": "neg_relation_absent_class", "cf_rel_neg_thing": "neg_relation_thing", "cf_pos_absent": "neg_position_absent_class",
        "count_pos": "count_exact", "pos_pos_sole": "pos_pos_sole"}
SOURCE = {"pos_pos_sole": "ground_v2_gap_rule", "cf": "v1.2_catfree", "neg13": "v1.3_negatives", "count": "v1.3_count_pairs",
          "zh": "v1.3_refcoco_zh"}


def q_out(q, kind):
    bx = q.get("boxes") or ([q["box"]] if q.get("box") else [])
    return dict(text=q["text"], lang=q["lang"], kind=kind, src=q.get("src", ""), present=bool(q["present"]),
                n_targets=int(q.get("n_targets", len(bx) if q["present"] else 0)), boxes=[[float(v) for v in b] for b in bx])


def main():
    ap = argparse.ArgumentParser()
    for k in ("gd_full", "catfree12", "neg13", "count", "zh", "pool_natural", "replay", "eval_ids", "commit", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/data", exist_ok=True); os.makedirs(f"{a.out}/manifest", exist_ok=True)
    held = set().union(*[set(v) for v in json.load(open(a.eval_ids)).values()])
    rows, st = [], collections.defaultdict(collections.Counter)

    def add(iid, file, W, H, qs, source):
        assert iid not in held, f"eval image {iid}"
        rows.append(dict(image_id=int(iid), file=file, width=int(W), height=int(H), questions=qs, source=source))
        for q in qs:
            k = q["kind"]
            st[k]["n"] += 1; st[k][q["lang"]] += 1; st[k]["pos" if q["present"] else "neg"] += 1; st[k]["_source"] = source

    # (a) pos_pos_sole, from the glance-ground-v2 conversion (src gap_rule)
    for fn in sorted(glob.glob(f"{a.gd_full}/*.jsonl")):
        per = collections.defaultdict(list); meta = {}
        for l in open(fn):
            r = json.loads(l)
            meta[r["image_id"]] = r
            per[r["image_id"]] += [q_out(q, "pos_pos_sole") for q in r["questions"] if q.get("src") == "gap_rule"]
        for iid, qs in per.items():
            if qs:
                r = meta[iid]
                add(iid, r["img"].split("/")[-1], r["wh"][0], r["wh"][1], qs, SOURCE["pos_pos_sole"])
    # (a) v1.2 category-free set and (b) v1.3 negatives
    for path, src in ((a.catfree12, SOURCE["cf"]), (a.neg13, SOURCE["neg13"])):
        for l in open(path):
            r = json.loads(l)
            qs = [q_out(q, KIND[q["qtype"]]) for q in r["questions"]]
            add(r["image_id"], r["img"].split("/")[-1], r["wh"][0], r["wh"][1], qs, src)
    # (b) count minimal pairs
    for l in open(a.count):
        r = json.loads(l)
        add(r["image_id"], r["img"].split("/")[-1], r["wh"][0], r["wh"][1], [q_out(q, "count_exact") for q in r["questions"]], SOURCE["count"])
    # (b) Chinese RefCOCO translations (zh only; English original referenced in the manifest)
    pool = {}
    for l in open(a.pool_natural):
        r = json.loads(l)
        pool.setdefault(r["image_id"], dict(r, questions=[]))["questions"] += r["questions"]
    with open(f"{a.out}/manifest/refcoco_zh_pairs.jsonl", "w") as mf:
        for l in open(a.zh):
            z = json.loads(l)
            p = pool.get(z["image_id"])
            en = next((q for q in (p or {}).get("questions", []) if q["text"] == z["en"] and q.get("box")), None)
            if en is None:
                continue
            q = dict(text=z["zh"], lang="zh", kind="refcoco_zh", src=en.get("src", "") + "_zh", present=True, n_targets=1,
                     boxes=[[float(v) for v in en["box"]]])
            add(z["image_id"], p["img"].split("/")[-1], p["wh"][0], p["wh"][1], [q], SOURCE["zh"])
            mf.write(json.dumps(dict(image_id=z["image_id"], box=en["box"], dataset=en.get("src"),
                                     en_sha1=hashlib.sha1(z["en"].encode()).hexdigest(), zh=z["zh"]), ensure_ascii=False) + "\n")
    # (b) replay manifest (ids only)
    n_rep = 0
    with open(f"{a.out}/manifest/refcoco_replay.jsonl", "w") as mf:
        for l in open(a.replay):
            r = json.loads(l)
            if r.get("src") != "natural":
                continue
            for q in r["questions"]:
                mf.write(json.dumps(dict(image_id=r["image_id"], present=q["present"], box=q.get("box"), dataset=q.get("src"),
                                         neg=q.get("neg"), text_sha1=hashlib.sha1(q["text"].encode()).hexdigest())) + "\n")
                n_rep += 1
    st["refcoco_replay_manifest"]["n"] = n_rep
    df = pd.DataFrame(rows)
    for s in sorted(df.source.unique()):
        df[df.source == s].to_parquet(f"{a.out}/data/train-{s}.parquet", index=False)
    out = {k: dict(v) for k, v in st.items()}
    json.dump(dict(commit=a.commit, rows=len(df), images=int(df.image_id.nunique()), per_kind=out), open(f"{a.out}/stats.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
