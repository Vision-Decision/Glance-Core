#!/usr/bin/env bash
# Push ablation code (+ needed data/prep files unless --code-only) to /content/dv on the VM.
set -euo pipefail
S=${1:?session}; MODE=${2:-}
cd "$(dirname "$0")/.."
export COPYFILE_DISABLE=1
TMP=$(mktemp -d)
if [ "$MODE" = "--code-only" ]; then
  tar czf "$TMP/abl.tgz" --exclude='__pycache__' ablation
else
  tar czf "$TMP/abl.tgz" --exclude='__pycache__' ablation scripts/colab_setup.sh data/prep/train.jsonl data/prep/train_neg05.jsonl \
    data/prep/eval_grefcoco_val_merged.jsonl data/prep/eval_refcoco_val.jsonl data/prep/eval_refcoco_testA.jsonl data/prep/eval_refcoco_testB.jsonl
fi
timeout 600 colab-pool upload "$S" "$TMP/abl.tgz" /content/abl.tgz
echo 'import subprocess;print(subprocess.run("mkdir -p /content/dv && tar xzf /content/abl.tgz -C /content/dv && ls /content/dv /content/dv/ablation",shell=True,capture_output=True,text=True))' | timeout 300 colab-pool exec "$S" --timeout 240
rm -rf "$TMP"
