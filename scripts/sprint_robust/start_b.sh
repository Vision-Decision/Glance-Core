#!/usr/bin/env bash
# Second VM: create (retrying), full setup, upload the augmented pool, start queue q_b, keep alive with a hard deadline.
S=${1:-sr-b}; Q=${2:-q_b}; HOURS=${3:-9}
cd "$(dirname "$0")/../.."
SP=<scratch>
scripts/sprint_robust/launch_vm.sh "$S" full 240 || exit 1
colab-pool upload "$S" $SP/aug_all_v1.jsonl /content/up/aug_natural.jsonl > /dev/null 2>&1
scripts/sprint_robust/rbg.sh "$S" "$Q"
echo "[start_b] $Q started on $S $(date -u)"
scripts/sprint_robust/keepalive.sh "$S" "$HOURS" > $SP/ka_$S.log 2>&1
