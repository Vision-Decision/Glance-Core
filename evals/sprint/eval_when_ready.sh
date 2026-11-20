#!/usr/bin/env bash
# eval_when_ready.sh HF_PATH TAG WAIT_LOG@PATTERN RUNS : poll the HF repo until HF_PATH exists, then run eval_ckpt.sh.
cd /content/dv
HP=$1; TAG=$2; WAIT=$3; RUNS=$4
D=/content/ckpt/$TAG; mkdir -p $D
until timeout 1800 hf download Vision-Decision/glance-train-ckpts-legacy $HP --local-dir $D >/dev/null 2>&1; do sleep 120; done
echo "[q] $HP available $(date -u +%T)"
bash evals/sprint/eval_ckpt.sh $HP $TAG $WAIT "$RUNS"
