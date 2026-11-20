#!/usr/bin/env bash
# Crop-and-zoom second prefills after a first pass. Usage: zoom_queue.sh CKPT FIRST_PREFIX BENCH "frac:max_vis:upscale:point ..." [WAIT_LOG]
# Waits for "ALL DONE" in WAIT_LOG (if given) so it never shares the GPU with another sweep.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=$1; F=$2; B=$3; SPECS=$4; WAIT=${5:-}
cd /content/dv
[ -n "$WAIT" ] && until grep -q "ALL DONE" $WAIT 2>/dev/null; do sleep 30; done
for s in $SPECS; do
  IFS=: read FR V U PT K <<< "$s"; K=${K:-1}
  O=${F}_zoom${FR}_${V}u${U}${PT}; [ $K -gt 1 ] && O=${O}c$K
  [ -f $O.info.json ] && continue
  echo "[q] $(date -u +%T) $O"
  python evals/sprint/infer.py --ckpt $CK --bench /content/gui/$B.jsonl --zoom_from $F --zoom_frac $FR --zoom_point $PT \
    --max_vis $V --upscale $U --zoom_cands $K --out $O --no_maps > $O.log 2>&1; grep -av Warning $O.log | tail -2
done
echo "[q] ALL DONE zoom $(date -u +%T)"
