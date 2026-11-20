#!/usr/bin/env bash
# Launch a job script detached on a pooled VM:  scripts/sprint_robust/rbg.sh SESSION JOB
# JOB = basename of scripts/sprint_robust/{JOB}.sh or scripts/sprint_robust/jobs/{JOB}.sh (uploaded first);
# log -> /content/sr/{JOB}.log on the VM.
S=${1:?session}; J=${2:?job}
cd "$(dirname "$0")/../.."
SRC=scripts/sprint_robust/$J.sh; [ -f "$SRC" ] || SRC=scripts/sprint_robust/jobs/$J.sh
[ -f "$SRC" ] || { echo "no job $J"; exit 1; }
colab-pool upload "$S" "$SRC" "/content/dv/$SRC" > /dev/null 2>&1
CMD="mkdir -p /content/sr; bash /content/dv/$SRC" LOG="/content/sr/$J.log" python3 scripts/remote_bg.py | colab-pool exec "$S" 2>&1 | grep -v '^\[pool\]'
