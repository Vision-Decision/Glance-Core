#!/usr/bin/env bash
# Robustness-eval VM setup (after scripts/sync.sh SESSION --code-only and the HF token upload).
# Deps + Qwen3.5-4B + v0/v1 ckpts + only the COCO train2014 images used by glance-robust-ref (+ eval prep files once the
# dataset exists). Image list: /content/img_urls.txt if uploaded, else derived from the dataset.
# Usage (VM): bash /content/dv/scripts/robust/vm_setup.sh > /content/rsetup.log 2>&1   (idempotent; re-run for the data step)
set -uo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
echo "[rsetup] $(date) start"
python -c "import fla" 2>/dev/null || { pip uninstall -y -q torchao || true
  pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate pyarrow pandas; }
python - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Qwen/Qwen3.5-4B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"])
for f in ["v0/ckpt_final_bf16.pt", "v1/ckpt_final.pt"]:
    print(hf_hub_download("Vision-Decision/glance-train-ckpts-legacy", f, local_dir="/content/ckpt"))
EOF
mkdir -p /content/data/coco/train2014
if python -c "from huggingface_hub import snapshot_download as s; s('Vision-Decision/glance-robust-ref', repo_type='dataset', local_dir='/content/robust_data')" 2>/dev/null; then
  (cd /content/dv && python scripts/robust/make_eval.py --data /content/robust_data/data --out /content/robust)
  [ -f /content/img_urls.txt ] || cat /content/robust_data/data/*.jsonl | python -c "import sys,json; print('\n'.join(sorted({json.loads(l)['file'] for l in sys.stdin})))" \
    | sed 's#^#http://images.cocodataset.org/train2014/#' > /content/img_urls.txt
else
  echo "[rsetup] dataset not on HF yet"
fi
cd /content/data/coco/train2014 && [ -f /content/img_urls.txt ] && xargs -P 32 -n 50 wget -q -nc < /content/img_urls.txt
echo "[rsetup] images $(ls /content/data/coco/train2014 | wc -l)"
echo "[rsetup] $(date) DONE"
