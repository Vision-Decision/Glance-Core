#!/usr/bin/env bash
# Missing control, arm B: upstream Glance-X trainer, Glance-X recipe and data mix, image_first batches only.
# Usage (VM): RUN=i2t_only_s0 SEED=0 bash /content/dv/scripts/glancex/vm_train_gx.sh > /content/gx/train_RUN.log 2>&1
# Everything else is configs/glance-x-4b-k8.json (Qwen3.5-4B-Base, K=8, 8 images/step, 2 epochs = 6,100 steps).
set -uo pipefail
RUN=${RUN:-i2t_only_s0}; SEED=${SEED:-0}; MP=${MP:-1,0,0}; REPO=Vision-Decision/glance-train-ckpts
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/content/gx/train/$RUN; mkdir -p $O
push() { timeout 1800 hf upload $REPO $O glancex/train/$RUN --repo-type model --include "*.json" --include "*.jsonl" --include "*.log" --include "ckpt_*.pt" \
           --commit-message "glancex train $RUN $1" >/dev/null 2>&1 && echo "[gx-train] pushed $1 $(date -u)"; }
( while sleep 1800; do [ -f $O/DONE ] && break; cp /content/gx/train_$RUN.log $O/train.log 2>/dev/null; push periodic; done ) &
RES=""; [ -f $O/ckpt_last.pt ] && RES="resume=$O/ckpt_last.pt"
cd /content/glance-x
python -m glancex.train --cfg configs/glance-x-4b-k8.json --set mode_probs=$MP seed=$SEED out_dir=$O \
   train_file=/content/gxdata/train_merged.jsonl image_root=/content/data/coco/train2014 $RES
touch $O/DONE; cp /content/gx/train_$RUN.log $O/train.log; push final
SRC=$O/ckpt_final.pt TAG=$RUN MODES="${EVAL_MODES:-image_first}" bash /content/dv/scripts/glancex/vm_eval.sh > /content/gx/eval_$RUN.log 2>&1
echo "[gx-train] DONE $(date -u)"
