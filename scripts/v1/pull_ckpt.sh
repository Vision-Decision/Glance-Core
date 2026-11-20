#!/usr/bin/env bash
# Keep dv-v1 alive while training runs detached, and pull ckpt_last.pt (or ckpt_final.pt) to checkpoints/v1/.
# Usage: S=dv-v1 R=/content/out/v1 scripts/v1/pull_ckpt.sh [pull_interval_seconds]
cd "$(dirname "$0")/../.."
S=${S:-dv-v1}; R=${R:-/content/out/v1}; L=checkpoints/v1; INT=${1:-1800}
mkdir -p "$L"; last=""
while true; do
  info=$(timeout 180 env RSH_TIMEOUT=120 scripts/rsh.sh $S "cd $R && for f in ckpt_final.pt ckpt_last.pt; do [ -f \$f ] && { echo \$f \$(stat -c %Y \$f); break; }; done; pgrep -f train.py >/dev/null && echo RUNNING || echo STOPPED; tail -1 train_log.jsonl | cut -c1-300" 2>/dev/null)
  echo "$(date '+%F %T') $(echo "$info" | tr '\n' ' ' | cut -c1-400)"
  f=$(echo "$info" | awk '/^ckpt_/{print $1; exit}'); m=$(echo "$info" | awk '/^ckpt_/{print $2; exit}')
  if [ -n "$f" ] && [ "$f:$m" != "$last" ]; then
    sleep 60
    if timeout 1200 colab-pool download $S "$R/$f" "$L/$f.tmp" >/dev/null 2>&1 && [ -s "$L/$f.tmp" ]; then
      [ "$f" = ckpt_final.pt ] || rm -f "$L"/ckpt_last.pt
      mv "$L/$f.tmp" "$L/$f"; last="$f:$m"
      echo "$(date '+%F %T') pulled $f ($(du -h "$L/$f" | cut -f1))"
    else
      rm -f "$L/$f.tmp"; echo "$(date '+%F %T') download of $f failed, will retry"
    fi
  fi
  if [ "$f" = ckpt_final.pt ] && [ "$last" = "$f:$m" ]; then echo "final checkpoint pulled, done"; exit 0; fi
  if echo "$info" | grep -q STOPPED && [ "$f" != ckpt_final.pt ]; then echo "$(date '+%F %T') train.py not running and no final ckpt"; exit 1; fi
  t=0; while [ $t -lt "$INT" ]; do
    sleep 300; t=$((t+300))
    timeout 90 env RSH_TIMEOUT=60 scripts/rsh.sh $S "echo ok" >/dev/null 2>&1
  done
done
