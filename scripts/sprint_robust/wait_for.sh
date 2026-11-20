#!/usr/bin/env bash
# Block until a VM log line matches PATTERN (checked every 2 min):  wait_for.sh SESSION "/content/sr/x.log" PATTERN [max_min]
S=${1:?}; LOG=${2:?}; P=${3:?}; M=${4:-240}
cd "$(dirname "$0")/../.."
end=$(( $(date +%s) + M * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  out=$(RSH_TIMEOUT=60 timeout 150 scripts/rsh.sh "$S" "grep -h -E '$P' $LOG 2>/dev/null | tail -3")
  if [ -n "$(echo "$out" | tr -d '[:space:]')" ]; then echo "$out"; exit 0; fi
  sleep 120
done
echo "timeout"; exit 1
