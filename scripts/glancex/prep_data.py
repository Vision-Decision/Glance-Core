"""Rebuild Glance-X's training file and the Glance-Data diagnostic set from the HF parquet release (CPU; gen-host).

  python scripts/glancex/prep_data.py --orig data/prep/train.jsonl --out ~/glancex/data [--revision SHA]

Glance-X trains on data/train.jsonl = merge(prepare_refcoco train.jsonl, Glance-Data aug_train_all.jsonl)
(Glance-X scripts/merge_data.py). Its prepare_refcoco.py is our dv/data/prep.py (identical code), so the original half
is our committed data/prep/train.jsonl (355,385 requests, 24,407 images). The augmentation half is only published as
parquet (Vision-Decision/Glance-Data, rows written by glance_data/export_hf.py); this script inverts that export
(boxes are rounded to 0.01 px there, the only lossy step) and then applies merge_data.py's logic verbatim.

Outputs ({out}/):
  aug_train_all.jsonl     Glance-Data train, one record per image (Glance-X record format)
  diagnostic.jsonl        Glance-Data diagnostic split (300 refcoco_val images), Glance-X record format (eval)
  train_merged.jsonl      Glance-X training file (orig=True on the original requests)
  train_dv_single.jsonl   the same mix in dv/data/dataset.py format for our recipe: requests with <= 1 target
                          (box = the target or null); multi-target requests dropped, as in v1.2's `single` view
  provenance.json         counts + source revisions
"""
import argparse, collections, glob, json, os

import pandas as pd
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--revision", default="0da8cf88053728f58674c7d5b8357db4740d3f62")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
d = snapshot_download("Vision-Decision/Glance-Data", repo_type="dataset", revision=a.revision,
                      local_dir=os.path.join(a.out, "_gd"))


def records(split):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{d}/data/{split}-*.parquet"))])
    for row in df.itertuples(index=False):
        qs = []
        for q in row.questions:
            boxes = [[float(v) for v in b] for b in q["boxes"]]
            qs.append(dict(text=q["text"], lang=q["lang"], kind=q["kind"], src=q["src"], present=bool(q["present"]),
                           n_targets=int(q["n_targets"]), boxes=boxes, box=boxes[0] if len(boxes) == 1 else None))
        yield dict(image_id=int(row.image_id), file=row.file, W=int(row.width), H=int(row.height), questions=qs)


stats = dict(glance_data_revision=a.revision)
for split, fn in (("train", "aug_train_all.jsonl"), ("diagnostic", "diagnostic.jsonl")):
    n_img = n_q = 0
    kinds = collections.Counter()
    with open(os.path.join(a.out, fn), "w") as f:
        for r in records(split):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_img += 1; n_q += len(r["questions"]); kinds.update(q["kind"] for q in r["questions"])
    stats[split] = dict(images=n_img, questions=n_q, kinds=dict(kinds))
    print(split, n_img, n_q, flush=True)

# ---- Glance-X scripts/merge_data.py (verbatim logic)
recs, by_img = {}, collections.defaultdict(list)
for line in open(a.orig):
    r = json.loads(line)
    recs.setdefault(r["image_id"], dict(image_id=r["image_id"], file=r["file"], W=r["W"], H=r["H"]))
    for q in r["questions"]:
        by_img[r["image_id"]].append(dict(q, orig=True, boxes=[q["box"]] if q.get("box") is not None else None,
                                          kind=q.get("src", "orig"), lang="en"))
n_aug = 0
for line in open(os.path.join(a.out, "aug_train_all.jsonl")):
    r = json.loads(line)
    if r["image_id"] not in recs:
        continue
    for q in r["questions"]:
        by_img[r["image_id"]].append(dict(q, orig=False))
        n_aug += 1
n_dv = n_multi = 0
with open(os.path.join(a.out, "train_merged.jsonl"), "w") as f, open(os.path.join(a.out, "train_dv_single.jsonl"), "w") as g:
    for iid, r in recs.items():
        f.write(json.dumps(dict(r, questions=by_img[iid]), ensure_ascii=False) + "\n")
        qs = []
        for q in by_img[iid]:
            bs = q.get("boxes") or ([q["box"]] if q.get("box") is not None else [])
            if q["present"] and len(bs) > 1:
                n_multi += 1
                continue
            qs.append(dict(text=q["text"], present=bool(q["present"]), box=bs[0] if (q["present"] and bs) else None,
                           src=q.get("src"), kind=q.get("kind"), lang=q.get("lang", "en"), orig=bool(q["orig"])))
        n_dv += len(qs)
        g.write(json.dumps(dict(r, questions=qs), ensure_ascii=False) + "\n")
n = sum(len(v) for v in by_img.values())
stats["merged"] = dict(images=len(recs), questions=n, orig=n - n_aug, aug=n_aug)
stats["dv_single"] = dict(images=len(recs), questions=n_dv, dropped_multi_target=n_multi)
json.dump(stats, open(os.path.join(a.out, "provenance.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in stats.items() if k not in ("train", "diagnostic")}))
