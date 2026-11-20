#!/usr/bin/env bash
# GUI sprint VM setup: deps, Qwen3.5-4B weights, GUI benchmarks, sprint checkpoints from the private HF repo.
# Usage (VM, detached): bash /content/dv/evals/sprint/setup_vm.sh [ckpt paths in Vision-Decision/glance-train-ckpts-legacy ...]
set -uo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1
cd /content/dv
echo "[setup] $(date -u) start"
pip uninstall -y -q torchao || true
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate pyarrow pandas scipy
python -c "import transformers,fla,peft;print(transformers.__version__,fla.__version__,peft.__version__)"
python -c "from huggingface_hub import snapshot_download as s;print(s('Qwen/Qwen3.5-4B',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.py']))"
mkdir -p /content/ckpt
for p in "${@:-v1/ckpt_final.pt}"; do
  timeout 1800 hf download Vision-Decision/glance-train-ckpts-legacy "$p" --local-dir /content/ckpt >/dev/null && echo "[setup] got $p"
done
[ -f /content/gui/sspro.jsonl ] || timeout 5400 python evals/gui_data.py --root /content/gui
ls -la /content/gui/*.jsonl
echo "[setup] $(date -u) DONE"
