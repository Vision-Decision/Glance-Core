#!/usr/bin/env bash
# Push this worktree's code (+ the small prepped files under the main checkout's data/: RefCOCO eval preps, the v1 natural
# pool / calib split) to a pooled Colab VM at /content/dv.   Usage: scripts/v12/sync.sh SESSION [--code-only]
set -euo pipefail
S=${1:?session}; MODE=${2:-}
cd "$(dirname "$0")/../.."
DATA=${DATA:-~/Developer/decision-vison/data}
export COPYFILE_DISABLE=1
TMP=$(mktemp -d)
tar czf "$TMP/dv.tgz" --no-xattrs --exclude='__pycache__' dv train.py eval.py scripts configs evals
if [ "$MODE" != "--code-only" ]; then
  tar czf "$TMP/data.tgz" --no-xattrs -C "$DATA/.." data/prep data/prep_v1/pool_natural.jsonl data/prep_v1/calib_natural.jsonl data/prep_v1/stats_natural.json
  colab-pool upload "$S" "$TMP/data.tgz" /content/dv_data.tgz
fi
colab-pool upload "$S" "$TMP/dv.tgz" /content/dv.tgz
echo 'import subprocess;print(subprocess.run("mkdir -p /content/dv && tar xzf /content/dv.tgz -C /content/dv && ([ -f /content/dv_data.tgz ] && tar xzf /content/dv_data.tgz -C /content/dv || true); ls /content/dv /content/dv/data",shell=True,capture_output=True,text=True))' | colab-pool exec "$S"
rm -rf "$TMP"
