"""Download Vision-Decision/Glance-Data (private; HF_TOKEN from the environment) and print a short profile.
  python scripts/ground_v2/fetch_glance_data.py --out ~/gv2/glance_data
"""
import argparse, collections, glob, json

import pandas as pd
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = snapshot_download("Vision-Decision/Glance-Data", repo_type="dataset", local_dir=a.out)
for split in ["train", "diagnostic"]:
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{d}/data/{split}-*.parquet"))])
    qs = [q for l in df.questions for q in l]
    print(split, "images", len(df), "questions", len(qs), "columns", list(df.columns))
    print(" kind x present", dict(collections.Counter((q["kind"], bool(q["present"])) for q in qs)))
    print(" src", dict(collections.Counter(q["src"] for q in qs)), "lang", dict(collections.Counter(q["lang"] for q in qs)))
    print(" n_targets", dict(collections.Counter(min(int(q["n_targets"]), 4) for q in qs)))
    ex = collections.defaultdict(list)
    for q in qs:
        if len(ex[q["kind"]]) < 4:
            ex[q["kind"]].append((q["lang"], q["text"], int(q["n_targets"])))
    print(" examples", json.dumps(ex, ensure_ascii=False))
    print(" first row keys", {k: type(v).__name__ for k, v in df.iloc[0].items()}, "q keys", list(qs[0].keys()), "box ex", qs[0]["boxes"])
