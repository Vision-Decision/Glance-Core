#!/usr/bin/env bash
# Pack code (+ prepped annotations unless --code-only) and push to a pooled Colab session, unpack at /content/dv.
# Usage: scripts/sync.sh SESSION [--code-only]
set -euo pipefail
S=${1:?session}; MODE=${2:-}
cd "$(dirname "$0")/.."
export COPYFILE_DISABLE=1
TMP=$(mktemp -d)
if [ "$MODE" = "--code-only" ]; then
  tar czf "$TMP/dv.tgz" --exclude='__pycache__' dv train.py eval.py baseline.py scripts configs README.md 2>/dev/null || \
  tar czf "$TMP/dv.tgz" --exclude='__pycache__' dv train.py eval.py scripts configs
else
  tar czf "$TMP/dv.tgz" --exclude='__pycache__' dv train.py eval.py $(ls baseline.py README.md 2>/dev/null) scripts configs data/prep
fi
ls -la "$TMP/dv.tgz"
colab-pool upload "$S" "$TMP/dv.tgz" /content/dv.tgz
echo 'import subprocess;print(subprocess.run("mkdir -p /content/dv && tar xzf /content/dv.tgz -C /content/dv 2>/dev/null; ls /content/dv",shell=True,capture_output=True,text=True).stdout)' | colab-pool exec "$S"
rm -rf "$TMP"
