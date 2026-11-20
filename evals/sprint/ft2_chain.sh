#!/usr/bin/env bash
# sg-train: after ft1 finishes, ft2 = ft1 + zoom-scale crops (5k crop records + 35% ft1-mixture replay), lr 5e-5.
cd /content/dv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ -f /content/out/sprint_ft1/ckpt_final.pt ]; do sleep 60; done
sleep 30
C=/content/data/prep_crops5k
python evals/sprint/make_crops.py --pool /content/data/prep_gui_v2/pool_gui.jsonl,/content/data/prep_gui_v2/pool_guisyn.jsonl \
  --replay /content/data/prep_gui_v2/train.jsonl --out $C --max_crops 5000 > /content/prep_crops5k.log 2>&1
echo "[train] crops ready $(date -u +%T) $(tail -1 /content/prep_crops5k.log)"
OUT=/content/out/sprint_ft2
nohup bash scripts/v1/push_ckpt_hf.sh $OUT Vision-Decision/glance-train-ckpts-legacy sprint_gui/ft2 900 > /content/push_ft2.log 2>&1 &
python train.py --cfg configs/sprint_gui_ft.json --set train_file=$C/train.jsonl out_dir=$OUT \
  init_ckpt=/content/out/sprint_ft1/ckpt_final.pt lr=5e-5 lr_heads=1.5e-4 warmup_steps=20 > /content/train_ft2.log 2>&1
echo "[train] ft2 rc=$? $(date -u +%T)"
