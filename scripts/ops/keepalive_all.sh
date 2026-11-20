#!/usr/bin/env bash
# Ping every live pooled Colab session every 5 minutes so Colab does not reclaim it as idle.
# Usage: scripts/ops/keepalive_all.sh   (runs until killed; logs to outputs/ops/keepalive.log)
cd "$(dirname "$0")/../.."
mkdir -p outputs/ops
while true; do
  for s in $(timeout 120 colab-pool list --json 2>/dev/null | python3 -c "import json,sys; print(' '.join(x['name'] for x in json.load(sys.stdin).get('sessions',[])))"); do
    if timeout 90 env RSH_TIMEOUT=60 scripts/rsh.sh "$s" "echo ok" 2>/dev/null | grep -q '^ok'; then r=ok; else r=FAIL; fi
    echo "$(date -u '+%F %T') $s $r" >> outputs/ops/keepalive.log
  done
  sleep 300
done
