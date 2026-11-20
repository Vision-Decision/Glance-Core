#!/usr/bin/env bash
# Push a run's checkpoints to HF with single-file uploads (hf upload FOLDER --include with several patterns did not upload):
# ckpt_last + opt_last + train_log every hour, ckpt_final once it exists.  Usage (VM): bash vm_push.sh OUT PATH_IN_REPO
OUT=${1:-/content/out/v1_2}; P=${2:-v1.2}; REPO=Vision-Decision/glance-train-ckpts; export HF_HUB_OFFLINE=0
up() { [ -f "$OUT/$1" ] && timeout 3600 hf upload $REPO "$OUT/$1" "$P/$1" --repo-type model > /dev/null 2>&1 && echo "[push] $1 $(date -u +%T)"; }
while [ ! -f $OUT/ckpt_final.pt ]; do
  up ckpt_last.pt; up opt_last.pt; up train_log.jsonl; up config.json
  for i in $(seq 120); do [ -f $OUT/ckpt_final.pt ] && break; sleep 30; done
done
sleep 30
up ckpt_final.pt; up train_log.jsonl; up config.json
echo "[push] final done $(date -u)"
