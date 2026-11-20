#!/usr/bin/env bash
# PREREG group D on the VM with the S2 sprint pipeline (scripts/sprint_abs): extended row dumps at native GUI resolution
# (8192 tokens, no upscale, G4), official VenusBench-GD evaluator with the faithful mapping, official-style GREC metrics,
# and the training-free calibration (Platt + EM prior correction x Platt kappa).
# Needs /content/sabs/data/grefcalib.jsonl (scripts/sprint_abs/make_grefcalib.py, uploaded).
# Usage (VM): nohup bash vm_post_s2.sh [CKPT] > /content/v12_post_s2.log 2>&1 &
set -u
C=${1:-/content/out/v1_2/ckpt_final.pt}; TAG=v1.2; S=/content/sabs
until [ -f $C ]; do sleep 60; done
mkdir -p /content/ckpt/$TAG $S/data
ln -sf $C /content/ckpt/$TAG/ckpt.pt
cp -n /content/data/prep_v1/calib_natural.jsonl $S/data/
cd /content/dv
DUMP_ARGS="--gui_max_vis 8192 --gui_upscale 1.0" bash scripts/sprint_abs/vm_run.sh $TAG $TAG/ckpt.pt > $S/run_$TAG.log 2>&1
echo "[post_s2] dump done $(date -u)"
TAG=$TAG bash scripts/sprint_abs/chain_venus.sh > $S/chain_venus_$TAG.log 2>&1
echo "[post_s2] venus done $(date -u)"
TAG=$TAG BOOT=500 REPS=100 bash scripts/sprint_abs/vm_analyze.sh
timeout 1800 hf upload Vision-Decision/glance-train-ckpts $S/an_$TAG v1.2/post_s2 --repo-type model --exclude "rows/*" > /dev/null 2>&1 && echo "[post_s2] pushed"
echo "[post_s2] ALL DONE $(date -u)"
