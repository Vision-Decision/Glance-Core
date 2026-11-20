#!/usr/bin/env bash
# Keep a pooled VM alive (ping every 4 min) with a hard wall-clock deadline, after which the VM is stopped.
# Stop early (and stop the VM): touch /tmp/sr_stop_SESSION.   Usage: scripts/sprint_robust/keepalive.sh SESSION HOURS
S=${1:?session}; H=${2:-8}
cd "$(dirname "$0")/../.."
end=$(( $(date +%s) + H * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  [ -f "/tmp/sr_stop_$S" ] && { echo "$(date -u +%T) stop flag"; break; }
  out=$(RSH_TIMEOUT=60 timeout 120 scripts/rsh.sh "$S" "tail -qn1 /content/sr/*.log 2>/dev/null | tail -4; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader" 2>&1 | tr '\n' ' ' | cut -c1-400)
  echo "$(date -u +%T) $out"
  sleep 240
done
echo "$(date -u +%T) deadline/stop reached -> stopping $S"
timeout 300 colab-pool stop "$S"
