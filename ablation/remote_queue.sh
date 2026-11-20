#!/usr/bin/env bash
# On the VM: train + eval a list of runs sequentially (idempotent: finished stages are skipped).
# bash ablation/remote_queue.sh run1 run2 ...      (cwd /content/dv)
cd /content/dv
export PYTHONUNBUFFERED=1
for R in "$@"; do
  O=/content/out/abl/$R
  mkdir -p $O
  if [ ! -f $O/ckpt_final.pt ]; then
    echo "[queue] train $R $(date)"
    python -m ablation.train_ab --cfg ablation/configs/$R.json > $O/train_stdout.log 2>&1 || { echo "[queue] TRAIN FAIL $R"; continue; }
  fi
  if [ ! -f $O/rows.json ]; then
    echo "[queue] eval $R $(date)"
    python -m ablation.eval_ab --ckpt $O/ckpt_final.pt --out $O/eval.json --rows $O/rows.json > $O/eval_stdout.log 2>&1 || echo "[queue] EVAL FAIL $R"
  fi
  echo "[queue] done $R $(date)"
done
echo "[queue] ALL DONE $(date)"
