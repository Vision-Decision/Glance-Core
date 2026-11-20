#!/usr/bin/env bash
# v1.2 on one Colab G4: environment, the v1.1 base pools (rebuilt exactly as v1.1: fetch.sh + fetch_v11.sh + prep chain),
# Glance-Data (glance-ground-v2 `single`), eval sets, and a separate vLLM venv for the category-free generator.
# The CPU/network part runs in the background while the GPU generates (scripts/v12/vm_gen.sh).
# Usage (VM): nohup bash /content/dv/scripts/v12/vm_setup.sh > /content/v12_setup.log 2>&1 &
set -uo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
cd /content/dv
echo "[v12setup] $(date -u) start"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /content | tail -1
# 1) vLLM venv (own torch; must not touch the training env) + generator weights, in the background (NO_VLLM=1 skips it)
[ -n "${NO_VLLM:-}" ] || (
  set -e
  pip install -q uv
  [ -x /content/venv_vllm/bin/python ] || uv venv /content/venv_vllm --python 3.12
  uv pip install --python /content/venv_vllm/bin/python vllm pillow huggingface_hub
  /content/venv_vllm/bin/python -c "import vllm; print('[v12setup] vllm', vllm.__version__)"
  /content/venv_vllm/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.8-27B-FP8'))"
  echo "[v12setup] VLLM_READY $(date -u)"
) > /content/v12_vllm.log 2>&1 &
# 2) training env + Qwen3.5-4B + COCO train2014 (scripts/colab_setup.sh)
bash scripts/colab_setup.sh > /content/setup.log 2>&1; echo "[v12setup] colab_setup rc=$? $(date -u)"; tail -2 /content/setup.log
# 3) v1.1 raw data, then the v1.1 prep chain (natural pool comes from the synced data/prep_v1)
mkdir -p /content/data/prep_v1 && cp -n /content/dv/data/prep_v1/* /content/data/prep_v1/
bash scripts/v1/fetch.sh > /content/fetch.log 2>&1; echo "[v12setup] fetch done $(date -u)"
bash scripts/v1/fetch_v11.sh > /content/fetch11.log 2>&1; echo "[v12setup] fetch11 done $(date -u)"
bash scripts/v1/prep_v11.sh > /content/prep_v11.log 2>&1; echo "[v12setup] prep_v11 done $(date -u)"
# 4) Glance-Data (single view + diagnostic), robust-ref, v1.1 checkpoint
python - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Vision-Decision/glance-ground-v2", repo_type="dataset", local_dir="/content/gd",
                  allow_patterns=["single/*", "diagnostic*.jsonl", "stats.json"])
snapshot_download("Vision-Decision/glance-robust-ref", repo_type="dataset", local_dir="/content/robust_raw", allow_patterns=["data/*"])
print(hf_hub_download("Vision-Decision/glance-train-ckpts-legacy", "v1.1/ckpt_final.pt", local_dir="/content/ckpt"))
EOF
python scripts/robust/make_eval.py --data /content/robust_raw/data --out /content/robust
echo "[v12setup] ALL DONE $(date -u)"; df -h /content | tail -1
