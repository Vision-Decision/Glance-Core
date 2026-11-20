"""Push the v1.2 training data built on the VM to the private org dataset Vision-Decision/glance-train-data (v1.2/).
  python scripts/v12/push_data.py --commit <git hash>"""
import argparse, os
from huggingface_hub import HfApi

ap = argparse.ArgumentParser(); ap.add_argument("--commit", required=True); a = ap.parse_args()
api = HfApi(); R = "Vision-Decision/glance-train-data"
api.create_repo(R, repo_type="dataset", private=True, exist_ok=True)
files = {"/content/data/prep_v12/train.jsonl": "v1.2/train.jsonl",
         "/content/data/prep_v12/train_base.jsonl": "v1.2/train_base.jsonl",
         "/content/data/prep_v12/stats_v12.json": "v1.2/stats_v12.json",
         "/content/data/prep_v12/stats_base.json": "v1.2/stats_base.json",
         "/content/v12/out/catfree.jsonl": "v1.2/catfree.jsonl", "/content/v12/out/stats.json": "v1.2/catfree_stats.json"}
for p in ["pool_text", "pool_gui", "pool_pixmo", "pool_natural", "calib_natural", "calib_text", "calib_text_hard", "calib_gui", "eval_hashes"]:
    ext = "json" if p == "eval_hashes" else "jsonl"
    files[f"/content/data/prep_v1/{p}.{ext}"] = f"v1.2/pools_v1.1/{p}.{ext}"
for p in sorted(os.listdir("/content/data/prep_v1")):
    if p.startswith("stats_") and p.endswith(".json"):
        files[f"/content/data/prep_v1/{p}"] = f"v1.2/pools_v1.1/{p}"
readme = f"""# glance-train-data / v1.2 (private)

Training data of Glance v1.2, built on the Colab VM `v12` from repo commit `{a.commit}` (branch worktree-agent-ab225d57487eeae40).

- `train.jsonl`: the v1.2 mixture actually trained on (335,815 records, 1.15M questions; first 3,800 steps of one shuffled pass).
- `train_base.jsonl`: its v1.1-replay part. `pools_v1.1/`: the rebuilt v1.1 question pools + calibration splits + eval hashes.
- `catfree.jsonl`: the category-free set (scripts/v12/catfree_*.py; Qwen3.8-27B-FP8 wording, blind-verified).

Build:
```
bash scripts/v12/vm_setup.sh            # fetch.sh + fetch_v11.sh + prep_v11.sh (v1.1 pools), Glance-Data, robust-ref
bash scripts/v12/vm_gen.sh; bash scripts/v12/vm_gen_rel.sh 24000   # category-free generation (vLLM)
python scripts/v12/catfree_build.py --scenes scenes.jsonl,scenes_rel.jsonl --gen gen.jsonl,gen_rel.jsonl --out_dir out
python scripts/v12/build_mix.py --prep /content/data/prep_v1 --gd /content/gd/single --catfree out/catfree.jsonl \\
    --eval_ids eval_image_ids.json --base_total 650000 --out /content/data/prep_v12
```
Image paths are relative to the data root (COCO train2014, GUI and pixmo images are not included). License: mixed sources,
research use only (see the source datasets; Glance-Data is CC BY-NC 4.0).
"""
open("/content/README_v12.md", "w").write(readme)
files["/content/README_v12.md"] = "v1.2/README.md"
for src, dst in files.items():
    if os.path.exists(src):
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=R, repo_type="dataset")
        print("uploaded", dst, os.path.getsize(src) // 2**20, "MB", flush=True)
    else:
        print("missing", src)
print("DONE")
