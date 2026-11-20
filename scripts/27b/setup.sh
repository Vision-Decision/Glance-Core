#!/usr/bin/env bash
# 27B environment setup on a fresh Colab >=80GB GPU VM (H100 80GB / A100 80GB / G4 96GB).
# Usage (on the VM): bash /content/27b/setup.sh > /content/27b_setup.log 2>&1
# Idempotent. Pins match the v0 agent's scripts/colab_setup.sh so the same training code runs.
# Env:
#   MODEL_ID   (default Qwen/Qwen3.8-27B)
#   HF_HOME    (default /content/hf; point it at /content/drive/MyDrive/hf to use a Drive cache
#               if Drive has been mounted by a human via `colab drivemount`)
set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
export HF_HOME="${HF_HOME:-/content/hf}"
export HF_XET_HIGH_PERFORMANCE=1

echo "[27b-setup] $(date) start  model=$MODEL_ID  HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /content | tail -1

# transformers>=5.9 has Qwen3.5/3.8 (model_type qwen3_5); fla gives fused Gated-DeltaNet
# kernels; kernels lets transformers pull the hub causal-conv1d kernel (else torch fallback).
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" \
  accelerate pyarrow pandas bitsandbytes kernels
# Colab image ships torchao 0.10, which peft>=0.18 rejects at LoRA injection time (needs >0.16).
pip uninstall -y -q torchao || true
python - <<'EOF'
import torch, transformers, fla, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "fla", fla.__version__, "peft", peft.__version__)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
print("fla chunk_gated_delta_rule OK")
EOF

# ~56 GB bf16 safetensors (18 shards). Skip if already complete.
python - <<EOF
import os
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_ID", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"], max_workers=16)
sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / 2**30
print("model at", p, f"{sz:.1f} GiB")
EOF
echo "[27b-setup] $(date) DONE"
