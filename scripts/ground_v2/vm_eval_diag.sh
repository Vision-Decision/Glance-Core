#!/usr/bin/env bash
# Colab G4: evaluate v1 and v1.1 on the Glance-Data diagnostic split (+ pos_pos_sole add-on).
# Needs /content/dv (scripts/sync.sh SESSION --code-only) and an HF token on the VM (~/.cache/huggingface/token).
# Usage (VM): nohup bash /content/dv/scripts/ground_v2/vm_eval_diag.sh > /content/diag.log 2>&1 &
# Results: /content/diag/{v1,v1.1}.json (+ rows) -> Vision-Decision/glance-ground-v2 eval/diag_<date>/
set -uo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
echo "[diag] $(date -u) start"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import fla" 2>/dev/null || { pip uninstall -y -q torchao || true
  pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate pyarrow pandas; }
python - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Qwen/Qwen3.5-4B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"])
for f in ["v1/ckpt_final.pt", "v1.1/ckpt_final.pt"]:
    print(hf_hub_download("Vision-Decision/glance-train-ckpts-legacy", f, local_dir="/content/ckpt"))
for f in ["diagnostic.jsonl", "diagnostic_pos_sole.jsonl"]:
    print(hf_hub_download("Vision-Decision/glance-ground-v2", f, repo_type="dataset", local_dir="/content/gd"))
EOF
mkdir -p /content/data/coco/train2014 /content/diag
cat /content/gd/diagnostic.jsonl | python -c "import sys,json; print('\n'.join(sorted({json.loads(l)['img'].split('/')[-1] for l in sys.stdin})))" \
  | sed 's#^#http://images.cocodataset.org/train2014/#' > /content/diag_urls.txt
(cd /content/data/coco/train2014 && xargs -P 32 -n 20 wget -q -nc < /content/diag_urls.txt)
echo "[diag] images $(ls /content/data/coco/train2014 | wc -l) / $(wc -l < /content/diag_urls.txt)"
cd /content/dv
for M in v1 v1.1; do
  python scripts/ground_v2/eval_diag.py --ckpt /content/ckpt/$M/ckpt_final.pt \
    --recs /content/gd/diagnostic.jsonl,/content/gd/diagnostic_pos_sole.jsonl \
    --out /content/diag/$M.json --rows /content/diag/${M}_rows.jsonl.gz
  echo "[diag] $M done $(date -u)"
done
python - <<'EOF'
import datetime
from huggingface_hub import HfApi
d = datetime.date.today().isoformat()
HfApi().upload_folder(folder_path="/content/diag", repo_id="Vision-Decision/glance-ground-v2", repo_type="dataset",
                      path_in_repo=f"eval/diag_{d}", commit_message="diagnostic eval v1 / v1.1")
print("[diag] pushed eval/diag_" + d)
EOF
echo "[diag] $(date -u) DONE"
