#!/usr/bin/env bash
# v1.2 category-free generation on the VM GPU (runs alongside vm_setup.sh's CPU/network work).
# Usage (VM): nohup bash /content/dv/scripts/v12/vm_gen.sh [LIMIT] > /content/v12_gen.log 2>&1 &
set -uo pipefail
LIMIT=${1:-0}
until grep -q VLLM_READY /content/v12_vllm.log 2>/dev/null; do sleep 30; done
until [ -f /content/data/coco/train2014/.done ]; do sleep 30; done
echo "[gen] start $(date -u)"
cd /content/dv/scripts/v12
HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING /content/venv_vllm/bin/python catfree_gen.py --scenes /content/v12/scenes.jsonl \
  --img_root /content/data/coco/train2014 --out /content/v12/gen.jsonl --limit $LIMIT
echo "[gen] exit rc=$? $(date -u)"
