#!/usr/bin/env bash
# VM side: experiments 2+3. Pull seed-0 AR checkpoints from the private HF repo, run beam search (+ refinement) on each.
# [TAG=_full] [NOSPLIT=1] [IPB=n] bash ablation/v2_beam.sh [STRIDE] [RUN ...]   (cwd /content/dv; STRIDE = every n-th image)
# TAG suffixes the output dir (/content/out/v2/beam$TAG/<run>); NOSPLIT=1 drops the refinement (split_k) configs.
cd /content/dv
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IPB=${IPB:-1}   # images per batch; 1 keeps peak memory low (GDN states are copied per beam) when sharing the GPU
STRIDE=${1:-1}; shift
RUNS=${@:-"ar_atomic100_neg25 ar_digits_neg25 ar_hier_neg25 ar_atomic1000_neg25"}
LOG=/content/out/v2_jobs.log
C=/content/ckpt; mkdir -p $C
for R in $RUNS; do
  O=/content/out/v2/beam${TAG:-}/$R; mkdir -p $O
  [ -f $O/beam_rows.json ] && continue
  [ -f $C/ablation/${R}_bf16.pt ] || timeout 1200 hf download Vision-Decision/glance-train-ckpts-legacy ablation/${R}_bf16.pt --local-dir $C > /dev/null 2>&1
  case $R in
    ar_atomic100*) EXTRA="--split_k 2,4,10";;
    ar_digits*)    EXTRA="--split_k 2,4";;
    *)             EXTRA="";;
  esac
  [ -n "${NOSPLIT:-}" ] && EXTRA=""
  echo "[beam] start $R stride=$STRIDE $(date)" >> $LOG
  python -m ablation.beam_ab --ckpt $C/ablation/${R}_bf16.pt --configs b4lp0,b8lp0,b4lp1,b8lp1 $EXTRA \
     --image_stride $STRIDE --ipb $IPB --check --out $O/beam.json --rows $O/beam_rows.json > $O/stdout.log 2>&1 \
     && echo "[beam] done $R $(date)" >> $LOG || echo "[beam] FAIL $R $(date)" >> $LOG
done
echo "[beam] ALL DONE $(date)" >> $LOG
