#!/usr/bin/env bash
# Exp 6c/6a-generative: hybrid (Qwen3.5-4B, 3:1 GDN:attention) vs pure-attention (Qwen3-VL-4B-Instruct) generative
# grounding with ONE prompt for both (the official VenusBench/Qwen3-VL point prompt, [x,y] in 0-1000, [-1,-1] if absent):
#   needle canvases N=8..512 (1600 visual tokens, fixed), and a fixed 400-question ScreenSpot-Pro subset at 1024 / 4096 /
#   16384 visual-token budgets (no upscale). Rows: /content/arch/gdn/gen/<model>_<set>.jsonl
cd /content/dv; O=/content/arch/gdn/gen; mkdir -p $O
python - <<'PY'
import json, random
ids = [q["meta"]["id"] for l in open("/content/gui/sspro.jsonl") for q in json.loads(l)["questions"]]
random.Random(0).shuffle(ids); json.dump(ids[:400], open("/content/arch/gdn/gen/sspro_ids400.json", "w"))
PY
for M in Qwen/Qwen3.5-4B Qwen/Qwen3-VL-4B-Instruct; do
  T=${M#Qwen/}
  for n in 8 16 32 64 128 256 512; do
    python evals/run_baseline.py --model $M --bench /content/needle/needle_$n.jsonl --prompt venus --max_vis_tokens 1600 \
      --max_upscale 1.0 --out $O/${T}_needle_$n.jsonl > $O/${T}_needle_$n.log 2>&1; tail -1 $O/${T}_needle_$n.log
  done
  for V in 1024 4096 16384; do
    python evals/run_baseline.py --model $M --bench /content/gui/sspro.jsonl --prompt venus --max_vis_tokens $V --max_upscale 1.0 \
      --ids $O/sspro_ids400.json --out $O/${T}_sspro_$V.jsonl > $O/${T}_sspro_$V.log 2>&1; tail -1 $O/${T}_sspro_$V.log
  done
done
echo GEN_DONE
