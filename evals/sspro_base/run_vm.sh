#!/usr/bin/env bash
# Base Qwen3.5-4B (generative, greedy, thinking off, official SS-Pro Qwen3.5 click prompt) on ScreenSpot-Pro at
# NATIVE resolution: same resize rule as Glance's v11f_sspro_16384u1 run (resize_dims, cap 16,384 visual tokens,
# max_upscale 1.0). Questions visited in a seeded random order (seed 0) so a run cut by the time limit is a random subset.
# Usage (VM, detached): bash /content/dv/evals/sspro_base/run_vm.sh [time_limit_s]
set -uo pipefail
TL=${1:-2700}
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
cd /content/dv
OUT=/content/out/sspro_base; mkdir -p $OUT
echo "[setup] $(date -u)"
pip uninstall -y -q torchao || true
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" accelerate
python -c "import transformers,fla;print(transformers.__version__,fla.__version__)"
python -c "from huggingface_hub import snapshot_download as s;print(s('Qwen/Qwen3.5-4B',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.py']))" &
[ -f /content/gui/sspro.jsonl ] || python evals/gui_data.py --root /content/gui --only sspro || exit 1
wait
echo "[run] $(date -u)"
python evals/run_baseline.py --bench /content/gui/sspro.jsonl --prompt click --max_vis_tokens 16384 --max_upscale 1.0 \
  --shuffle_seed 0 --time_limit $TL --out $OUT/base_sspro_16384u1.jsonl
echo "[done] $(date -u) rows=$(wc -l < $OUT/base_sspro_16384u1.jsonl)"
