#!/usr/bin/env bash
# VM side, experiment 5 on an otherwise idle G4: (0) incremental-decoding equivalence check, (1) ours + AR-digits
# (incremental, presence-only) in our engine, (2) vLLM generative base with prefix caching, (3) vLLM without it.
cd /content/dv
export PYTHONUNBUFFERED=1
LOG=/content/out/v2_jobs.log
C=/content/ckpt; O=/content/out/v2/latency; mkdir -p $C $O /content/out/v2/beam_check
for R in ours_neg25 ar_digits_neg25; do
  [ -f $C/ablation/${R}_bf16.pt ] || timeout 1200 hf download Vision-Decision/glance-train-ckpts-legacy ablation/${R}_bf16.pt --local-dir $C > /dev/null 2>&1
done
echo "[lat] check $(date)" >> $LOG
timeout 1800 python -m ablation.beam_ab --ckpt $C/ablation/ar_digits_neg25_bf16.pt --configs b4lp0 --limit_images 24 --check \
  --out /content/out/v2/beam_check/ar_digits_neg25.json --rows /content/out/v2/beam_check/ar_digits_neg25_rows.json \
  > /content/out/v2/beam_check/ar_digits_neg25.log 2>&1 || echo "[lat] check FAIL" >> $LOG
echo "[lat] ours+ar $(date)" >> $LOG
timeout 5400 python -m ablation.latency_v2 --ours $C/ablation/ours_neg25_bf16.pt --ar $C/ablation/ar_digits_neg25_bf16.pt \
  --merge_lora --out $O/ours_ar.json > $O/ours_ar.log 2>&1 || echo "[lat] ours+ar FAIL" >> $LOG
echo "[lat] vllm $(date)" >> $LOG
timeout 5400 /content/vllm_env/bin/python ablation/latency_vllm.py --out $O/vllm_pc.json > $O/vllm_pc.log 2>&1 || echo "[lat] vllm FAIL" >> $LOG
echo "[lat] vllm-nopc $(date)" >> $LOG
timeout 5400 /content/vllm_env/bin/python ablation/latency_vllm.py --no_prefix_caching --strategies simul --out $O/vllm_nopc.json > $O/vllm_nopc.log 2>&1 || echo "[lat] vllm nopc FAIL" >> $LOG
echo "[lat] ALL DONE $(date)" >> $LOG
