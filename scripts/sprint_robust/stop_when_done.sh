#!/usr/bin/env bash
# Stop a VM as soon as its queue finishes (pattern in a VM log), via keepalive's stop flag plus a direct stop.
# Usage: stop_when_done.sh SESSION LOG PATTERN [max_min]
S=${1:?}; LOG=${2:?}; P=${3:?}; M=${4:-600}
cd "$(dirname "$0")/../.."
scripts/sprint_robust/wait_for.sh "$S" "$LOG" "$P" "$M"
touch "/tmp/sr_stop_$S"
echo "$(date -u +%T) $S queue done -> stopping"
timeout 300 colab-pool stop "$S"
