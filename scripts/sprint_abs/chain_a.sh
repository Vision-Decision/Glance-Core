#!/usr/bin/env bash
# sabs-a chain: wait for the v1 dump pipeline; re-dump the GREC sets with suppression peaks (multi-target decoding);
# then hard-negative fine-tune (A), then A + proper-scoring kappa (B).
cd /content/dv
export PYTHONPATH=/content/dv
until grep -q "\[sabs\].*DONE" /content/sabs/run_v1.log 2>/dev/null; do sleep 30; done
python scripts/sprint_abs/dump.py --ckpt /content/ckpt/v1/ckpt.pt --out /content/sabs/rows_v1p --calib_dir /content/sabs/data \
  --prep_dir /content/dv/data/prep --sets grefcalib,grec_val,grec_testA,grec_testB > /content/sabs/dump_v1p.log 2>&1
timeout 1800 hf upload Vision-Decision/glance-train-ckpts-legacy /content/sabs/rows_v1p sprint_abs/rows_v1p --repo-type model >/dev/null 2>&1
RUN=hnft STEPS=${STEPS:-300} bash scripts/sprint_abs/vm_train.sh > /content/sabs/train_hnft.log 2>&1
RUN=hnft_kp STEPS=${STEPS:-300} EXTRA="w_iou_brier=1.0 iou_head_on_sel=true" bash scripts/sprint_abs/vm_train.sh > /content/sabs/train_hnft_kp.log 2>&1
echo "[chain_a] DONE $(date -u)" > /content/sabs/chain_a.done
