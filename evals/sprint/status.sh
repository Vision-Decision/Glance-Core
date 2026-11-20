#!/usr/bin/env bash
# Poll sprint VMs (also keeps them alive): prints new "[q]"/"[done]"/"[ft]"/"[chain]"/error lines once.
#   evals/sprint/status.sh "sg-inf1 sg-train" [interval_s]
cd "$(dirname "$0")/../.."
VMS=$1; INT=${2:-240}; SEEN=${TMPDIR:-/tmp}/sg_seen.txt; touch $SEEN
while true; do
  for v in $VMS; do
    RSH_WALL=200 RSH_TIMEOUT=60 scripts/rsh.sh $v "cat /content/q_*.log /content/chain*.log /content/fetch_train.log /content/prep_train.log /content/train*.log /content/post*.log 2>/dev/null | grep -aE '^\[q\]|^\[done\]|^\[ft\]|^\[chain\]|^\[train\]|Traceback|Error|error:|Killed|out of memory|\"step\": [0-9]*00,' | tail -40" 2>/dev/null \
      | grep -v '^$' | while read -r l; do k="$v|$l"; grep -qxF "$k" $SEEN || { echo "$k" >> $SEEN; echo "$v: ${l:0:300}"; }; done
  done
  sleep $INT
done
