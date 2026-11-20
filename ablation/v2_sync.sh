#!/usr/bin/env bash
# Push ablation code + small data/prep manifests (+ the HF token, for pulling/pushing private ckpts) to /content/dv.
# ablation/v2_sync.sh SESSION [--code-only]
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
  cp ~/.cache/huggingface/token "$TMP/hf_token"
  timeout 300 colab-pool upload "$S" "$TMP/hf_token" /content/hf_token
fi
timeout 600 colab-pool upload "$S" "$TMP/abl.tgz" /content/abl.tgz
echo 'import subprocess;print(subprocess.run("mkdir -p /content/dv /root/.cache/huggingface && (test -f /content/hf_token && cp /content/hf_token /root/.cache/huggingface/token; true) && tar xzf /content/abl.tgz -C /content/dv && ls /content/dv",shell=True,capture_output=True,text=True))' | timeout 300 colab-pool exec "$S" --timeout 240 | grep -v '^\[pool\]'
rm -rf "$TMP"
