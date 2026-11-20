#!/usr/bin/env bash
# One-shot environment + data setup on a fresh Colab GPU VM.
# Usage (on the VM): bash /content/dv/scripts/colab_setup.sh > /content/setup.log 2>&1
# Idempotent: re-running skips finished steps.
set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "[setup] $(date) start"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"

# --- python deps -----------------------------------------------------------
# transformers>=5.9 for Qwen3.5; flash-linear-attention gives the fused Gated-DeltaNet
# kernels (chunk_gated_delta_rule). causal-conv1d is NOT built: our custom forward uses
# torch depthwise conv1d (k=4), which is cheap; see README "decisions".
# Colab preinstalls torchao 0.10 which makes peft (>=0.21) raise on LoRA injection; we do not need it.
pip uninstall -y -q torchao || true
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate pyarrow pandas
python - <<'EOF'
import transformers, fla, peft, torch
print("transformers", transformers.__version__, "fla", fla.__version__, "peft", peft.__version__)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
print("fla chunk_gated_delta_rule OK")
EOF

# --- model weights ---------------------------------------------------------
python - <<'EOF'
from huggingface_hub import snapshot_download
for rid in ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base"]:
    p = snapshot_download(rid, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"])
    print("downloaded", rid, p)
EOF

# --- COCO train2014 images -------------------------------------------------
mkdir -p /content/data/coco
cd /content/data/coco
if [ ! -f train2014/.done ]; then
  if [ ! -f train2014.zip ]; then
    wget -q http://images.cocodataset.org/zips/train2014.zip
  fi
  unzip -q -o train2014.zip
  rm -f train2014.zip
  touch train2014/.done
fi
echo "[setup] images: $(ls train2014 | wc -l)"
echo "[setup] $(date) DONE"
