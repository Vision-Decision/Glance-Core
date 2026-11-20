#!/usr/bin/env bash
# 27B runs using the v0 code UNMODIFIED (only config/CLI overrides). Run on the VM after
#   scripts/sync.sh SESSION   and   bash /content/dv/scripts/colab_setup.sh   (+ pip uninstall -y torchao).
# Usage: bash /content/27b/train_27b.sh smoke|full [extra key=value overrides...]
# Launch detached from the laptop:
#   CMD="bash /content/27b/train_27b.sh full" LOG=/content/out/v0_27b.log python3 scripts/remote_bg.py | colab-pool exec SESSION
set -euo pipefail
cd /content/dv
export HF_HOME="${HF_HOME:-/content/hf}"           # 27B weights cached here by scripts/27b/setup.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODE=${1:?smoke|full}; shift
# bf16 base (measured: 51 GB weights, ~61 GB peak at 4x1.1k tokens), LoRA r=64 like v0,
# same effective batch as v0 (8 images/opt step), lower LoRA lr for the larger model.
COMMON="model_id=Qwen/Qwen3.8-27B quant=bf16 lora_r=64 lora_alpha=128 max_vis_tokens=640 lr=1e-4"
case $MODE in
  smoke) python train.py --cfg configs/v0_4b.json --set $COMMON images_per_step=4 grad_accum=1 \
           limit_records=50 max_steps=100 warmup_steps=10 log_every=10 out_dir=/content/out/overfit_27b "$@" ;;
  full)  python train.py --cfg configs/v0_4b.json --set $COMMON images_per_step=4 grad_accum=2 \
           epochs=1 save_every=250 out_dir=/content/out/v0_27b "$@" ;;
esac
