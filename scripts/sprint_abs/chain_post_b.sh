#!/usr/bin/env bash
# sabs-b, after the multi-target run: finish the VenusBench dumps of both retrained variants (the 8-worker loader had
# exhausted /dev/shm on the large GUI prefixes), fetch v1 / hnft rows from the private repo, analyse both variants and
# compare each with v1 on identical items.
cd /content/dv
export PYTHONPATH=/content/dv PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/content/sabs; REPO=Vision-Decision/glance-train-ckpts-legacy
until grep -q "^\[dump\] grec_testB" $S/train_hnft_mt.log 2>/dev/null; do sleep 60; done
pkill -f "sprint_abs/dump.py"; sleep 10; rm -f /dev/shm/torch_* 2>/dev/null
until [ -f $S/chain_mt.done ]; do sleep 20; done
for RUN in hnft_mt hnft; do
  if [ $RUN = hnft ]; then
    [ -f /content/out/hnft/ckpt_final.pt ] || { hf download $REPO sprint_abs/hnft/ckpt_final.pt --repo-type model --local-dir /content/dl >/dev/null && mkdir -p /content/out/hnft && mv /content/dl/sprint_abs/hnft/ckpt_final.pt /content/out/hnft/; }
    hf download $REPO --repo-type model --include "sprint_abs/rows_hnft/*" --local-dir /content/dl >/dev/null
    mkdir -p $S/rows_hnft && cp -n /content/dl/sprint_abs/rows_hnft/*.jsonl.gz $S/rows_hnft/
  fi
  python scripts/sprint_abs/dump.py --ckpt /content/out/$RUN/ckpt_final.pt --out $S/rows_$RUN --calib_dir $S/data \
    --prep_dir /content/dv/data/prep --sets ss2,venus > $S/dump_venus_$RUN.log 2>&1
  timeout 1800 hf upload $REPO $S/rows_$RUN sprint_abs/rows_$RUN --repo-type model >/dev/null 2>&1
  echo "[train] $(date -u) DONE (post)" >> $S/train_$RUN.log
done
hf download $REPO --repo-type model --include "sprint_abs/rows_v1/*" "sprint_abs/rows_v1p/*" --local-dir /content/dl >/dev/null
mkdir -p $S/rows_v1 $S/rows_v1p
cp -n /content/dl/sprint_abs/rows_v1/*.jsonl.gz $S/rows_v1/; cp -n /content/dl/sprint_abs/rows_v1p/*.jsonl.gz $S/rows_v1p/
for RUN in hnft hnft_mt; do RUN=$RUN bash scripts/sprint_abs/chain_an_retrain.sh; done
echo "[post_b] DONE $(date -u)" > $S/post_b.done
