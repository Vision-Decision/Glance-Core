"""dHash every ScreenSpot-v2 / ScreenSpot-Pro / VenusBench-GD image (fetched with evals/gui_data.py) -> eval_hashes.json.
Same hash + index as dv/data/prep_v1.py (phash_file, HashIndex max_d=3).
  python evalhash.py --gui /data/guisyn/evalgui --out /data/guisyn/eval_hashes.json"""
import argparse, json, os, sys
sys.path.insert(0, "/data/guisyn/repo")
from dv.data.prep_v1 import _hash_many

ap = argparse.ArgumentParser(); ap.add_argument("--gui"); ap.add_argument("--out")
a = ap.parse_args()
out = {}
for name in ["ss2", "sspro", "venus"]:
    recs = [json.loads(l) for l in open(f"{a.gui}/{name}.jsonl")]
    out[name] = [h for h in _hash_many([os.path.join(r["image_root"], r["file"]) for r in recs]) if h]
    # ScreenSpot-Pro apps (for the overlap note)
    if name == "sspro":
        from collections import Counter
        print("sspro apps", Counter(q["meta"]["application"] for r in recs for q in r["questions"]).most_common())
json.dump(out, open(a.out, "w"))
# second stage (confirms coarse hits): 32x32 grayscale thumbnails of every eval image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_shards import thumb32
paths = []
for name in ["ss2", "sspro", "venus"]:
    paths += [(name, os.path.join(r["image_root"], r["file"])) for r in map(json.loads, open(f"{a.gui}/{name}.jsonl"))]
with ThreadPoolExecutor(24) as ex:
    th = list(ex.map(lambda p: thumb32(p[1]), paths))
ok = [i for i, t in enumerate(th) if t is not None]
np.savez_compressed(a.out.replace(".json", "_thumbs.npz"), thumbs=np.stack([th[i] for i in ok]),
                    names=np.array([paths[i][0] + ":" + os.path.basename(paths[i][1]) for i in ok]))
print("thumbs", len(ok))
