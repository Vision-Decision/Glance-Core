#!/usr/bin/env bash
# Push code + data/prep + the HF token (never printed) to a pooled Colab session; unpack at /content/dv.
# Usage: scripts/glancex/sync.sh SESSION
set -euo pipefail
S=${1:?session}
cd "$(dirname "$0")/../.."
export COPYFILE_DISABLE=1
TMP=$(mktemp -d)
tar czf "$TMP/dv.tgz" --exclude='__pycache__' dv train.py eval.py scripts/glancex scripts/sprint_abs scripts/colab_setup.sh configs data/prep/eval_*.jsonl
cp ~/.cache/huggingface/token "$TMP/hf_token"
colab-pool upload "$S" "$TMP/hf_token" /content/hf_token >/dev/null
colab-pool upload "$S" "$TMP/dv.tgz" /content/dv.tgz >/dev/null
echo 'import subprocess;print(subprocess.run("mkdir -p /content/dv /content/gx /root/.cache/huggingface && cp /content/hf_token /root/.cache/huggingface/token && rm /content/hf_token && tar xzf /content/dv.tgz -C /content/dv && ls /content/dv",shell=True,capture_output=True,text=True).stdout)' | colab-pool exec "$S"
rm -rf "$TMP"
