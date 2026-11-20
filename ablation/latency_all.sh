#!/usr/bin/env bash
# On the VM, GPU otherwise idle: latency for every finished run.
cd /content/dv
for R in "$@"; do
  O=/content/out/abl/$R
  [ -f $O/ckpt_final.pt ] || continue
  python -m ablation.eval_ab --ckpt $O/ckpt_final.pt --no_eval --latency --out $O/latency.json > $O/latency_stdout.log 2>&1
  echo "[lat] $R $(date)"
done
echo "[lat] ALL DONE"
