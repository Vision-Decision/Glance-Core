#!/usr/bin/env bash
# arch-exp VM setup (Colab G4). Usage (VM, detached):
#   bash /content/dv/arch/vm/setup.sh "coco gui" "Vision-Decision/glance-train-ckpts-legacy:v1.1/ckpt_final.pt Vision-Decision/glance-train-ckpts:v1.2/ckpt_final.pt"
# Needs /root/.cache/huggingface/token (uploaded, never printed). Idempotent.
set -uo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1
ASSETS=${1:-"coco"}; CKPTS=${2:-""}
cd /content/dv
echo "[setup] $(date -u +%T) start"
pip uninstall -y -q torchao || true
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate pyarrow pandas scipy scikit-learn
python -c "import transformers,fla,peft;print('[setup]',transformers.__version__,fla.__version__,peft.__version__)"
python -c "from huggingface_hub import snapshot_download as s;print(s('Qwen/Qwen3.5-4B',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.py']))"
mkdir -p /content/ckpt
for spec in $CKPTS; do
  repo=${spec%%:*}; path=${spec#*:}
  timeout 1800 hf download $repo "$path" --local-dir /content/ckpt/${repo##*/} >/dev/null && echo "[setup] got $spec"
done
if [[ " $ASSETS " == *" coco "* ]]; then
  mkdir -p /content/data/coco && cd /content/data/coco
  if [ ! -f train2014/.done ]; then
    wget -q http://images.cocodataset.org/zips/train2014.zip && unzip -q -o train2014.zip && rm -f train2014.zip && touch train2014/.done
  fi
  echo "[setup] coco images: $(ls train2014 | wc -l)"; cd /content/dv
fi
if [[ " $ASSETS " == *" gui "* ]]; then
  [ -f /content/gui/sspro.jsonl ] || timeout 5400 python evals/gui_data.py --root /content/gui --only sspro
  ls -la /content/gui/*.jsonl
fi
if [[ " $ASSETS " == *" venus "* ]]; then
  [ -f /content/gui/venus.jsonl ] || timeout 5400 python evals/gui_data.py --root /content/gui --only venus
fi
echo "[setup] $(date -u +%T) DONE"
