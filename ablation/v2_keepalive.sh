#!/usr/bin/env bash
# Local babysitter for one VM: ping every 5 min (hard timeout on each call), print new queue lines,
# pull results every 3rd ping.  Runs until killed or until the file /tmp/v2_stop_<S> exists.
# ablation/v2_keepalive.sh SESSION
S=${1:?session}
cd "$(dirname "$0")/.."
n=0; seen=""
while [ ! -f /tmp/v2_stop_$S ]; do
  o=$(RSH_WALL=150 RSH_TIMEOUT=120 scripts/rsh.sh $S "echo ok; cat /content/out/v2_queue.log /content/out/v2_jobs.log 2>/dev/null; tail -c 300 /content/out/v2_launch.log 2>/dev/null; for f in /content/out/abl/*/train_stdout.log; do echo \$f \$(tail -1 \$f | grep -o '\"step\": [0-9]*'); done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>&1)
  echo "$o" | grep -q "^ok" || echo "[keepalive $S] ping failed $(date +%H:%M)"
  q=$(echo "$o" | grep "^\[")
  new=$(comm -13 <(echo "$seen" | sort) <(echo "$q" | sort)); [ -n "$new" ] && echo "$new"
  seen="$q"
  n=$((n+1))
  if [ $((n % 3)) -eq 0 ]; then ablation/v2_pull.sh $S | tail -1; echo "$o" | grep "step\|%" | tr '\n' ' '; echo; fi
  sleep 300
done
echo "[keepalive $S] stopped"
