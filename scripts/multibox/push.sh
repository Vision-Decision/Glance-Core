#!/usr/bin/env bash
# Pack code + eval sets + pilot data and start scripts/multibox/vm_run.sh MODE detached on a pooled session.
# Usage: scripts/multibox/push.sh SESSION MODES [DATA_DIR]   (MODES e.g. joint  or  base,heads)   (DATA_DIR holds prep_mb/ and prep_v1/calib_natural.jsonl)
set -euo pipefail
S=${1:?session}; MODE=${2:?mode}; DATA=${3:-../../data}
cd "$(dirname "$0")/../.."
export COPYFILE_DISABLE=1
TMP=$(mktemp -d)
mkdir -p "$TMP/stage/data/prep_mb" "$TMP/stage/data/prep_v1"
cp "$DATA"/prep_mb/*.jsonl "$TMP/stage/data/prep_mb/"; cp "$DATA"/prep_v1/calib_natural.jsonl "$TMP/stage/data/prep_v1/"
tar czf "$TMP/dv.tgz" --exclude='__pycache__' dv train.py eval.py scripts configs data/prep -C "$TMP/stage" data/prep_mb data/prep_v1
ls -la "$TMP/dv.tgz"
colab-pool upload "$S" "$TMP/dv.tgz" /content/dv.tgz
[ -f ~/.cache/huggingface/token ] && colab-pool upload "$S" ~/.cache/huggingface/token /content/.hf_token  # private ckpt repo
CMD="mkdir -p /content/dv && tar xzf /content/dv.tgz -C /content/dv && bash /content/dv/scripts/multibox/vm_run.sh $MODE" \
  LOG=/content/mb.log python3 scripts/remote_bg.py | colab-pool exec "$S"
rm -rf "$TMP"
