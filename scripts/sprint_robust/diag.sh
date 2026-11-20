#!/usr/bin/env bash
# Diagnosis on the description-shift eval set (runs while vm_setup.sh is still fetching training pools):
#   v1 (Glance heads), base Qwen3.5-4B generative, and the v1 LoRA through the generative pathway.
# Usage (VM): bash /content/dv/scripts/sprint_robust/diag.sh > /content/sr/diag.log 2>&1
cd /content/dv
export PYTHONPATH=/content/dv TQDM_DISABLE=1
until [ -f /content/data/coco/train2014/.done ] && [ -f /content/ckpt/v1/ckpt_final.pt ]; do sleep 30; done
mkdir -p /content/sr /content/data/prep_v1
cp -n /content/up/calib_natural.jsonl /content/data/prep_v1/ 2>/dev/null
D=/content/out/sr/diag; mkdir -p $D
echo "[diag] v1 $(date -u)"
python eval.py --ckpt /content/ckpt/v1/ckpt_final.pt --calib_dir /content/data/prep_v1 --prep_dir /content/dv/data/prep \
  --sets rg --rows_dir $D/v1 --out $D/v1_eval.json > $D/v1_eval.log 2>&1
echo "[diag] base_gen $(date -u)"
python scripts/sprint_robust/base_gen.py --files /content/dv/data/prep/eval_rg.jsonl --out_dir $D/base > $D/base.log 2>&1
echo "[diag] v1-lora gen $(date -u)"
python scripts/sprint_robust/base_gen.py --files /content/dv/data/prep/eval_rg.jsonl --lora_ckpt /content/ckpt/v1/ckpt_final.pt \
  --out_dir $D/v1gen > $D/v1gen.log 2>&1
echo "[diag] DONE $(date -u)"
