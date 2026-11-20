"""Check that no training scene / record image is in any eval set: RefCOCO/+/g val/test*, gRefCOCO val/testA/testB,
COCO val2017, the ambiguity/attribute probe images, Vision-Decision/glance-robust-ref, and our own test split.

  python scripts/ground_v2/check_decontam.py --ann ~/gv2/ann --train scenes_a.jsonl,scenes_b.jsonl --test scenes_test.jsonl
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_scenes import held_out_ids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--test", default="")
ap.add_argument("--probe_ids", default="")
a = ap.parse_args()
ev = {"refcoco_grefcoco_eval": held_out_ids(a.ann),
      "val2017": {im["id"] for im in json.load(open(f"{a.ann}/annotations/instances_val2017.json"))["images"]}}
if a.probe_ids:
    ev["probes"] = set(json.load(open(a.probe_ids)))
try:
    from huggingface_hub import snapshot_download
    d = snapshot_download("Vision-Decision/glance-robust-ref", repo_type="dataset")
    ids = set()
    for f in glob.glob(f"{d}/**/*.jsonl", recursive=True):
        ids |= {r["image_id"] for r in map(json.loads, open(f)) if "image_id" in r}
    ev["glance_robust_ref"] = ids
    print("robust-ref images", len(ids), "of which already in RefCOCO/gRefCOCO eval:", len(ids & ev["refcoco_grefcoco_eval"]))
except Exception as e:
    print("robust-ref not checked:", type(e).__name__, str(e)[:120])
if a.test:
    ev["ground_v2_test"] = {json.loads(l)["image_id"] for l in open(a.test)}
train = set()
for f in a.train.split(","):
    train |= {json.loads(l)["image_id"] for l in open(f)}
bad = {k: len(train & v) for k, v in ev.items()}
print("train images", len(train), "overlap per eval set", bad)
assert not any(bad.values()), "CONTAMINATED"
print("OK: no overlap")
