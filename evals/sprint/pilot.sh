#!/usr/bin/env bash
# Pilot on sg-train while GroundCUA downloads: mixture from what exists, 60 steps from v1.1 ckpt_last (memory/speed).
cd /content/dv
P=/content/data/prep_gui_pilot; mkdir -p $P
cp /content/data/prep_gui/pool_guisyn.jsonl $P/ 2>/dev/null
python evals/sprint/prep_train.py --out $P --total_q 60000 > /content/prep_pilot.log 2>&1 || { echo "[chain] pilot prep FAIL"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py --cfg configs/sprint_gui_ft.json --set train_file=$P/train.jsonl out_dir=/content/out/pilot \
  init_ckpt=/content/ckpt/v1.1/ckpt_last.pt max_steps=60 log_every=5 save_every_min=0 > /content/train_pilot.log 2>&1
echo "[chain] pilot rc=$? $(date -u +%T)"
