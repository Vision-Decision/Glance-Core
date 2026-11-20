#!/usr/bin/env bash
# Wait for the 27B eval on dv-27b-2, pull results, keep the VM alive meanwhile.
cd "$(dirname "$0")/../.."
S=dv-27b-2; while true; do
  out=$(RSH_TIMEOUT=90 scripts/rsh.sh $S "test -f /content/out/v0_27b/eval_results.json && echo RESULTS_''READY; pgrep -f eval.py >/dev/null && echo EVALRUN; tail -c 300 /content/out/v0_27b_eval.log" 2>&1)
  if echo "$out" | grep -q RESULTS_READY; then
    colab-pool download $S /content/out/v0_27b/eval_results.json outputs/27b/eval_results.json >/dev/null 2>&1
    colab-pool download $S /content/out/v0_27b_eval.log outputs/27b/eval.log >/dev/null 2>&1
    colab-pool download $S /content/out/v0_27b/train_log.jsonl outputs/27b/train_log.jsonl >/dev/null 2>&1
    echo "done"; exit 0; fi
  if ! echo "$out" | grep -q EVALRUN && echo "$out" | grep -qi "error\|traceback"; then echo "eval failed:"; echo "$out" | tail -5; exit 1; fi
  sleep 240
done
