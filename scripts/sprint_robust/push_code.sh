#!/usr/bin/env bash
# Push code + eval prep files to a VM without disturbing running bash scripts: extract to a temp dir, then copy with
# --remove-destination (new inodes; a running bash keeps reading the old file).   Usage: push_code.sh SESSION
S=${1:?}
cd "$(dirname "$0")/../.."
T=/tmp/sr_push_$$.tgz
COPYFILE_DISABLE=1 tar czf $T --exclude='__pycache__' dv train.py eval.py baseline.py scripts configs evals \
  data/prep/eval_rg.jsonl data/prep/eval_robust_*.jsonl
colab-pool upload "$S" $T /content/push.tgz > /dev/null 2>&1
scripts/rsh.sh "$S" "rm -rf /content/push_x && mkdir -p /content/push_x && tar xzf /content/push.tgz -C /content/push_x 2>/dev/null && cp -a --remove-destination /content/push_x/. /content/dv/ && echo pushed"
rm -f $T
