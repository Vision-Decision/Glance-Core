#!/usr/bin/env bash
# Push code (+ data/prep) and the HF token to a pooled VM:  arch/vm/sync.sh SESSION [--code-only]
set -euo pipefail
S=${1:?session}; MODE=${2:-}
cd "$(dirname "$0")/../.."
export COPYFILE_DISABLE=1
TMP=$(mktemp -d)
X=$(ls -d dv train.py eval.py scripts configs arch ablation evals scan speed 2>/dev/null | tr "\n" " " || true)   # scan/speed live in glance-engine now (already on old VMs)
[ "$MODE" = "--code-only" ] || X="$X data/prep"
tar czf "$TMP/dv.tgz" --exclude='__pycache__' --exclude='*.npz' --exclude='outputs' $X
colab-pool upload "$S" "$TMP/dv.tgz" /content/dv.tgz >/dev/null
if [ "$MODE" != "--code-only" ]; then
  colab-pool upload "$S" "$HOME/.cache/huggingface/token" /content/.hftoken >/dev/null
fi
echo 'import subprocess;print(subprocess.run("mkdir -p /content/dv /root/.cache/huggingface; [ -f /content/.hftoken ] && cp /content/.hftoken /root/.cache/huggingface/token; tar xzf /content/dv.tgz -C /content/dv 2>/dev/null; ls /content/dv | tr \"\\n\" \" \"",shell=True,capture_output=True,text=True).stdout)' | colab-pool exec "$S" 2>&1 | grep -av '^\[pool\]'
rm -rf "$TMP"
