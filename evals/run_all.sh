#!/usr/bin/env bash
# Cross-domain eval of v0 on a Colab G4 VM. Code at /content/dv (v0 snapshot + evals/), ckpt at /content/ckpt.pt.
# Usage (detached):  bash /content/dv/evals/run_all.sh [setup|data|ours|baseline|all]
set -uo pipefail
STAGE=${1:-all}
cd /content/dv
OUT=/content/out/xdomain; mkdir -p $OUT
G=/content/gui
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false

if [[ $STAGE == setup || $STAGE == all ]]; then
  echo "[setup] $(date)"
  pip uninstall -y -q torchao || true
  pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate
  python -c "import transformers,fla,peft;print(transformers.__version__,fla.__version__,peft.__version__)"
  python -c "from huggingface_hub import snapshot_download as s;print(s('Qwen/Qwen3.5-4B',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.py']))"
fi
if [[ $STAGE == data || $STAGE == all ]]; then
  echo "[data] $(date)"
  python evals/gui_data.py --root $G || exit 1
fi
if [[ $STAGE == ours || $STAGE == all ]]; then
  for B in ss2 sspro venus; do
    for T in 640 4000; do
      echo "[ours] $B $T $(date)"
      LAT=0; [[ $B != venus ]] && LAT=50
      python evals/run_ours.py --ckpt /content/ckpt.pt --bench $G/$B.jsonl --max_vis_tokens $T --latency $LAT \
        --out $OUT/ours_${B}_${T}.jsonl
    done
  done
fi
if [[ $STAGE == baseline || $STAGE == all ]]; then
  echo "[baseline] $(date)"
  python evals/run_baseline.py --bench $G/ss2.jsonl --prompt click --max_vis_tokens 4000 --out $OUT/base_ss2_4000.jsonl
  python evals/run_baseline.py --bench $G/sspro.jsonl --prompt click --max_vis_tokens 4000 --out $OUT/base_sspro_4000.jsonl
  python evals/run_baseline.py --bench $G/venus.jsonl --prompt venus --max_vis_tokens 4000 --out $OUT/base_venus_4000.jsonl
fi
echo "[done] $STAGE $(date)"
