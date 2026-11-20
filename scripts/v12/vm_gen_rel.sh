#!/usr/bin/env bash
# Stop the main generation once it has N images (enough for the 150k-positive target), then run the relation-only scenes
# (scenes_rel.jsonl: relations are the scarcest kind). Usage (VM): nohup bash vm_gen_rel.sh 19000 > /content/v12_gen_rel.log 2>&1 &
N=${1:-19000}
until [ "$(wc -l < /content/v12/gen.jsonl)" -ge "$N" ] || ! pgrep -f "catfree_gen.py --scenes /content/v12/scenes.jsonl" > /dev/null; do sleep 30; done
pkill -f "catfree_gen.py --scenes /content/v12/scenes.jsonl"; sleep 5; pkill -9 -f "VLLM::EngineCore"; sleep 15  # the engine core is a separate process
echo "[gen_rel] main stopped at $(wc -l < /content/v12/gen.jsonl) images $(date -u)"
cd /content/dv/scripts/v12
HF_HUB_OFFLINE=1 /content/venv_vllm/bin/python catfree_gen.py --scenes /content/v12/scenes_rel.jsonl \
  --img_root /content/data/coco/train2014 --out /content/v12/gen_rel.jsonl
echo "[gen_rel] exit rc=$? $(date -u)"
