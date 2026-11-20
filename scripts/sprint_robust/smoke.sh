#!/usr/bin/env bash
# Smoke test of the new training paths (group sampling, consistency loss, caption LM loss): 6 steps, natural aug only.
cd /content/dv
export PYTHONPATH=/content/dv TQDM_DISABLE=1
until [ -f /content/data/coco/train2014/.done ] && [ -f /content/ckpt/v1/ckpt_final.pt ]; do sleep 30; done
O=/content/out/sr/smoke; mkdir -p $O; : > $O/empty.jsonl
python scripts/sprint_robust/mix_robust.py --v1_train $O/empty.jsonl --variants zh,para,func,vague,ufunc,ugeneric,absfunc,absgeneric \
  --out $O/mix.jsonl && head -c 300 $O/mix.jsonl && echo
python train.py --cfg configs/v1_4b.json --set train_file=$O/mix.jsonl out_dir=$O init_ckpt=/content/ckpt/v1/ckpt_final.pt \
  max_steps=6 warmup_steps=2 log_every=1 limit_records=200 w_cons=1.0 w_lm=0.2 lm_file=/content/up/lm_captions.jsonl lm_bs=4
echo "[smoke] DONE rc=$?"
