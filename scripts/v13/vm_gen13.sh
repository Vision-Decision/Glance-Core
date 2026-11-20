#!/usr/bin/env bash
# v1.3 generation VM (A100 or G4): vLLM venv, COCO train2014, then (1) Chinese translations of RefCOCO train expressions,
# (2) wording + blind verification of the v1.3 negative kinds (scenes_v13). Outputs pushed to Vision-Decision/glance-train-data.
# Usage (VM): nohup bash /content/dv/scripts/v13/vm_gen13.sh > /content/v13_gen.log 2>&1 &
set -uo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p /content/v13 /content/data/coco && cd /content/data/coco
( [ -f train2014/.done ] || { wget -q http://images.cocodataset.org/zips/train2014.zip && unzip -q -o train2014.zip && rm -f train2014.zip && touch train2014/.done; } ) &
bash /content/dv/scripts/v12/vm_vllm.sh > /content/v12_vllm.log 2>&1 || { echo "[g13] vllm setup failed"; exit 1; }
echo "[g13] vllm ready $(date -u)"
cd /content/dv/scripts/v13
MEM=${GPU_MEM:-0.92}; LEN=${MAX_LEN:-6144}
HF_HUB_OFFLINE=1 /content/venv_vllm/bin/python translate_zh.py --pool /content/dv/data/prep_v1/pool_natural.jsonl --out /content/v13/zh_trans.jsonl \
  --n 70000 --gpu_mem $MEM --max_model_len $LEN --max_num_seqs ${SEQS:-256}
echo "[g13] translate rc=$? $(date -u)"
wait
cd /content/dv/scripts/v12
HF_HUB_OFFLINE=1 /content/venv_vllm/bin/python catfree_gen.py --scenes /content/v13/scenes_v13_10k.jsonl --img_root /content/data/coco/train2014 \
  --out /content/v13/gen_v13.jsonl --n_neg 2 --gpu_mem $MEM --max_model_len $LEN --max_num_seqs ${SEQS:-256}
echo "[g13] gen rc=$? $(date -u)"
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for f in ["zh_trans.jsonl", "scenes_v13_10k.jsonl", "gen_v13.jsonl"]:
    api.upload_file(path_or_fileobj=f"/content/v13/{f}", path_in_repo=f"v1.3/raw/{f}", repo_id="Vision-Decision/glance-train-data", repo_type="dataset")
    print("[g13] pushed", f)
EOF
echo "[g13] ALL DONE $(date -u)"
