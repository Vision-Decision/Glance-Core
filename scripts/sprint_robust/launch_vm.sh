#!/usr/bin/env bash
# Create a G4 VM (retrying over accounts until capacity frees up), push code/data, start vm_setup.sh detached.
# Usage: scripts/sprint_robust/launch_vm.sh NAME [full|eval] [max_wait_min]
set -uo pipefail
S=${1:?name}; MODE=${2:-full}; MAXW=${3:-240}
cd "$(dirname "$0")/../.."
MAIN=~/Developer/decision-vison
t0=$(date +%s)
while true; do
  for acc in acct-0 acct-8 acct-7; do
    if timeout 900 colab-pool new "$S" --gpu g4 --account $acc --allow-other-account > /dev/null 2>&1; then
      echo "[launch] $S created on $acc $(date -u)"; break 2
    fi
  done
  [ $(( $(date +%s) - t0 )) -gt $(( MAXW * 60 )) ] && { echo "[launch] gave up after $MAXW min"; exit 1; }
  sleep 180
done
scripts/sync.sh "$S" > /dev/null
scripts/rsh.sh "$S" "mkdir -p /root/.cache/huggingface /content/up /content/sr" > /dev/null
COPYFILE_DISABLE=1 tar czf /tmp/sr_evals_$$.tgz --exclude=__pycache__ evals && colab-pool upload "$S" /tmp/sr_evals_$$.tgz /content/evals.tgz > /dev/null
colab-pool upload "$S" ~/.cache/huggingface/token /root/.cache/huggingface/token > /dev/null
if [ "$MODE" = "full" ]; then
  colab-pool upload "$S" $MAIN/data/prep_v1/pool_natural.jsonl /content/up/pool_natural.jsonl > /dev/null
fi
colab-pool upload "$S" $MAIN/data/prep_v1/calib_natural.jsonl /content/up/calib_natural.jsonl > /dev/null
colab-pool upload "$S" outputs/sprint_robust/data/lm_captions.jsonl /content/up/lm_captions.jsonl > /dev/null
CMD="MODE=$MODE bash /content/dv/scripts/sprint_robust/vm_setup.sh" LOG=/content/sr_setup.log python3 scripts/remote_bg.py | colab-pool exec "$S"
echo "[launch] setup started $(date -u)"
