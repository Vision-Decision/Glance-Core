#!/usr/bin/env bash
# VM setup for scripts/leak4b/run.py: deps, Glance v1.1 release (merged bf16 + heads), 20 COCO images.
# expects /content/dv_root (repo subset: dv/, scripts/leak4b/) and /root/.cache/huggingface/token
set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
pip uninstall -y -q torchao || true
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate safetensors
python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Vision-Decision/glance-train-ckpts-legacy", allow_patterns=["v1.1/release/dv_config.json", "v1.1/release/heads.safetensors",
                      "v1.1/release/merged-bf16/*"], local_dir="/content/rel", max_workers=8)
print("release at", p)
EOF
mkdir -p /content/imgs
python - <<'EOF'
import json, urllib.request, concurrent.futures as cf
recs = [json.loads(l) for l in open("/content/dv_root/scripts/leak4b/refcoco_val_20x8.jsonl")]
def get(f):
    urllib.request.urlretrieve(f"http://images.cocodataset.org/train2014/{f}", f"/content/imgs/{f}")
    return f
with cf.ThreadPoolExecutor(8) as ex:
    print(len(list(ex.map(get, [r["img"] for r in recs]))), "images")
EOF
python -c "import transformers, fla, torch; print(transformers.__version__, fla.__version__, torch.__version__, torch.cuda.get_device_name())"
echo SETUP_DONE
