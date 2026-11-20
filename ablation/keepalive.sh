#!/usr/bin/env bash
# Local babysitter: ping every 5 min (prints new queue.log lines), pull results every 4th ping. Stops on "ALL DONE".
cd "$(dirname "$0")/.."
seen=""; n=0
while true; do
  o=$(RSH_WALL=120 RSH_TIMEOUT=100 scripts/rsh.sh dv-abl "echo ok; cat /content/out/queue.log /content/out/post.log 2>/dev/null; tail -1 /content/out/abl/*/train_stdout.log 2>/dev/null | grep -o '\"step\": [0-9]*' | tail -1" 2>&1)
  if ! echo "$o" | grep -q "^ok"; then echo "[keepalive] ping failed $(date +%H:%M)"; fi
  q=$(echo "$o" | grep "^\[queue\]\|^\[post\]")
  new=$(comm -13 <(echo "$seen" | sort) <(echo "$q" | sort)); [ -n "$new" ] && echo "$new"
  seen="$q"
  n=$((n+1)); if [ $((n % 4)) -eq 0 ]; then ablation/pull_results.sh dv-abl | tail -1; echo "$o" | grep '"step"' | tail -1; fi
  echo "$q" | grep -q "post\] ALL DONE" && { ablation/pull_results.sh dv-abl | tail -1; echo "[keepalive] queue finished"; exit 0; }
  sleep 300
done
