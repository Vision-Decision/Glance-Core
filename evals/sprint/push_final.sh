#!/usr/bin/env bash
# push_final.sh LOCAL_CKPT HF_PATH : one upload attempt with a hard timeout; result goes to stdout (no secrets logged).
timeout 1800 hf upload Vision-Decision/glance-train-ckpts-legacy "$1" "$2" --repo-type model --commit-message "$2 $(date -u +%FT%T)" 2>&1 | tail -2
echo "[push] rc=${PIPESTATUS[0]} $2 $(date -u +%T)"
