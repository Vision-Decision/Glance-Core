#!/usr/bin/env bash
# v1.2 end-to-end chain on the VM once generation and the base prep are done:
# catfree_build -> build_mix -> train -> post-training eval (groups A-C, then D). One GPU job at a time.
# Usage (VM): nohup bash /content/dv/scripts/v12/vm_chain.sh > /content/v12_chain.log 2>&1 &
set -uo pipefail
cd /content/dv; export PYTHONPATH=/content/dv
until grep -q "\[gen_rel\] exit rc=0" /content/v12_gen_rel.log 2>/dev/null; do sleep 60; done
until grep -q "\[v12setup\] ALL DONE" /content/v12_setup.log 2>/dev/null; do sleep 60; done
echo "[chain] build $(date -u)"
python scripts/v12/catfree_build.py --scenes /content/v12/scenes.jsonl,/content/v12/scenes_rel.jsonl \
  --gen /content/v12/gen.jsonl,/content/v12/gen_rel.jsonl --out_dir /content/v12/out > /content/v12/catfree_build.log 2>&1 || exit 1
python scripts/v12/build_mix.py --prep /content/data/prep_v1 --gd /content/gd/single --catfree /content/v12/out/catfree.jsonl \
  --eval_ids /content/v12/eval_image_ids.json --base_total 650000 --out /content/data/prep_v12 > /content/v12/build_mix.log 2>&1 || { echo "[chain] build_mix FAILED"; exit 1; }
echo "[chain] data ready $(date -u)"; wc -l /content/data/prep_v12/train.jsonl
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for f, p in [("/content/v12/out/stats.json", "catfree_v12/catfree_stats.json"), ("/content/data/prep_v12/stats_v12.json", "catfree_v12/stats_v12.json"),
             ("/content/data/prep_v12/stats_base.json", "catfree_v12/stats_base.json"), ("/content/v12/gen.jsonl", "catfree_v12/gen.jsonl"),
             ("/content/v12/gen_rel.jsonl", "catfree_v12/gen_rel.jsonl"), ("/content/v12/out/catfree.jsonl", "catfree_v12/catfree.jsonl")]:
    try:
        api.upload_file(path_or_fileobj=f, path_in_repo=p, repo_id="Vision-Decision/glance-ground-v2", repo_type="dataset")
    except Exception as e:
        print("upload failed", f, type(e).__name__)
EOF
[ -f /content/v12/HOLD ] && { echo "[chain] HOLD present, waiting"; while [ -f /content/v12/HOLD ]; do sleep 30; done; }
pkill -9 -f "VLLM::EngineCore"; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader
bash scripts/v12/vm_train.sh > /content/v12_train.log 2>&1
echo "[chain] train finished $(date -u)"
bash scripts/v12/vm_post.sh > /content/v12_post.log 2>&1
bash scripts/v12/vm_post_s2.sh > /content/v12_post_s2.log 2>&1
echo "[chain] ALL DONE $(date -u)"
