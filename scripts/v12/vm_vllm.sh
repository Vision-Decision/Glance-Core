#!/usr/bin/env bash
# vLLM venv + generator weights (also part of vm_setup.sh)
  set -e
pip install -q uv
[ -x /content/venv_vllm/bin/python ] || uv venv /content/venv_vllm --python 3.12
uv pip install --python /content/venv_vllm/bin/python vllm pillow huggingface_hub
/content/venv_vllm/bin/python -c "import vllm; print('[v12setup] vllm', vllm.__version__)"
/content/venv_vllm/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.8-27B-FP8'))"
echo "[v12setup] VLLM_READY $(date -u)"
