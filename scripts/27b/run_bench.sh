#!/usr/bin/env bash
# Phase-1 driver (on VM): setup, then all bench configs; each config is a fresh process.
# Usage: nohup bash /content/27b/run_bench.sh > /content/27b_out/run.log 2>&1 &
set -uo pipefail
mkdir -p /content/27b_out
export HF_HOME="${HF_HOME:-/content/hf}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/content/27b/bench_27b.py
O=/content/27b_out
[ -f $O/setup.done ] || { bash /content/27b/setup.sh > $O/setup.log 2>&1 && touch $O/setup.done || { echo "SETUP FAILED"; exit 1; }; }
echo "== bf16 block-mask mbs 1,2,4"
python $B --mode throughput --quant bf16 --mbs 1,2,4 --maskmode block --out $O/tp_bf16_block.json
echo "== bf16 causal mbs 1,4 (baseline: sdpa flash path)"
python $B --mode throughput --quant bf16 --mbs 1,4 --maskmode causal --out $O/tp_bf16_causal.json
echo "== bf16 block-mask mbs 8"
python $B --mode throughput --quant bf16 --mbs 8 --maskmode block --out $O/tp_bf16_block_mb8.json
echo "== mask test sdpa,flex"
python $B --mode masktest --attn sdpa,flex --out $O/masktest.json
echo "ALL DONE"
