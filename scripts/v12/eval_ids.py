"""COCO image ids of every natural-image eval set we report, for build_mix.py's decontamination assert:
RefCOCO/+/g val/test*, gRefCOCO val/testA/testB (ground_v2.build_scenes.held_out_ids), COCO val2017, the ambiguity /
attribute probe images, glance-robust-ref, and the Glance-Data diagnostic split.

  python scripts/v12/eval_ids.py --ann ~/gv2/ann --gd ~/gv2/glance_data --probe_ids ~/gv2/probe_ids.json \
      --robust_ids robust_ids.json --out eval_image_ids.json
"""
import argparse, json, os, sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ground_v2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from build_scenes import held_out_ids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True); ap.add_argument("--gd", required=True)
ap.add_argument("--probe_ids", required=True); ap.add_argument("--robust_ids", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
ev = {"refcoco_grefcoco_eval": sorted(held_out_ids(a.ann)),
      "val2017": sorted(im["id"] for im in json.load(open(f"{a.ann}/annotations/instances_val2017.json"))["images"]),
      "probes": sorted(set(json.load(open(a.probe_ids)))),
      "glance_robust_ref": sorted(set(json.load(open(a.robust_ids)))),
      "glance_data_diagnostic": sorted(set(int(x) for x in pd.read_parquet(f"{a.gd}/data/diagnostic-00000-of-00001.parquet",
                                                                           columns=["image_id"]).image_id))}
json.dump(ev, open(a.out, "w"))
print({k: len(v) for k, v in ev.items()})
