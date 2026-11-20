#!/usr/bin/env bash
# Periodically pull the 27B run's latest checkpoint to outputs/27b/ckpt/, replacing the previous copy.
# Usage: S=SESSION scripts/27b/pull_ckpt.sh [interval_seconds]
# Also pings the VM every 5 min so Colab does not reclaim it as idle. Every remote call has a hard timeout
# so a wedged exec cannot hold the pool lock.
cd "$(dirname "$0")/../.."
S=${S:-dv-27b}; R=/content/out/v0_27b; L=outputs/27b/ckpt; INT=${1:-2400}
mkdir -p "$L"; last=""
while true; do
  info=$(timeout 180 env RSH_TIMEOUT=120 scripts/rsh.sh $S "cd $R && for f in ckpt_final.pt ckpt_last.pt; do [ -f \$f ] && { echo \$f \$(stat -c %Y \$f); break; }; done; pgrep -f train.py >/dev/null && echo RUNNING || echo STOPPED" 2>/dev/null)
  f=$(echo "$info" | awk '/^ckpt_/{print $1; exit}'); m=$(echo "$info" | awk '/^ckpt_/{print $2; exit}')
  running=$(echo "$info" | grep -c RUNNING)
  if [ -n "$f" ] && [ "$f:$m" != "$last" ]; then
    # wait until the file stops growing (save in progress)
    sleep 60
    if timeout 900 colab-pool download $S "$R/$f" "$L/$f.tmp" >/dev/null 2>&1 && [ -s "$L/$f.tmp" ]; then
      rm -f "$L"/ckpt_*.pt; mv "$L/$f.tmp" "$L/$f"; last="$f:$m"
      echo "$(date '+%F %T') pulled $f ($(du -h "$L/$f" | cut -f1))"
    else
      rm -f "$L/$f.tmp"; echo "$(date '+%F %T') download of $f failed, will retry"
    fi
  fi
  if [ "$f" = ckpt_final.pt ] && [ "$last" = "$f:$m" ]; then echo "final checkpoint pulled, done"; exit 0; fi
  if [ "$running" = 0 ] && [ -z "$info" ]; then echo "$(date '+%F %T') VM unreachable"; fi
  # "not found" can be transient (account switching under the pool lock); trust only reconcile
  if echo "$info" | grep -q "not found" && timeout 180 colab-pool reconcile 2>&1 | grep -q "$S: gone from server"; then
    echo "$(date '+%F %T') session $S gone from server (confirmed by reconcile)"; exit 2; fi
  if echo "$info" | grep -qx STOPPED && [ "$f" != ckpt_final.pt ]; then echo "$(date '+%F %T') train.py not running and no final ckpt — check the run"; exit 1; fi
  # sleep INT in 5-minute chunks, pinging the VM in between (keep-alive)
  t=0; while [ $t -lt "$INT" ]; do
    sleep 300; t=$((t+300))
    timeout 90 env RSH_TIMEOUT=60 scripts/rsh.sh $S "echo ok" >/dev/null 2>&1
  done
done
