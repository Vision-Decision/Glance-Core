#!/usr/bin/env bash
# Inference sweep for one checkpoint. Usage: bash queue_inf.sh CKPT TAG "bench:max_vis:upscale ..." [extra infer args]
# Writes /content/sprint/{TAG}_{bench}_{max_vis}u{upscale}.{rows.jsonl.gz,maps.npz,info.json}; skips finished runs.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=$1; TAG=$2; RUNS=$3; shift 3
cd /content/dv; mkdir -p /content/sprint
for spec in $RUNS; do
  IFS=: read B V U <<< "$spec"
  O=/content/sprint/${TAG}_${B}_${V}u${U}
  [ -f $O.info.json ] && { echo "[q] skip $O"; continue; }
  echo "[q] $(date -u +%T) $O"
  python evals/sprint/infer.py --ckpt $CK --bench /content/gui/$B.jsonl --max_vis $V --upscale $U --out $O "$@" > $O.log 2>&1; grep -av Warning $O.log | tail -3
done
echo "[q] ALL DONE $TAG $(date -u +%T)"
