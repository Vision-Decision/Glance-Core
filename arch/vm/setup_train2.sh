#!/usr/bin/env bash
# Phase-2 training VM from scratch: env + Qwen3.5-4B + COCO + SS-Pro + v1.1 ckpt, the v1.1 raw GUI/text/pixmo data
# (fetch.sh + fetch_v11.sh), and the identical filtered mixture from HF (arch/phase2/train_base_avail.jsonl).
set -uo pipefail
cd /content/dv
bash arch/vm/setup.sh "coco gui" "Vision-Decision/glance-train-ckpts-legacy:v1.1/ckpt_final.pt"
bash scripts/v1/fetch.sh > /content/logs/fetch.log 2>&1; echo "[st2] fetch $(date -u +%T)"
bash scripts/v1/fetch_v11.sh > /content/logs/fetch11.log 2>&1; echo "[st2] fetch11 $(date -u +%T)"
mkdir -p /content/data/prep_arch
python - <<'PY'
import json, os, shutil
from huggingface_hub import hf_hub_download
for f in ["train_base_avail.jsonl", "gui_replay.jsonl"]:
    try:
        p = hf_hub_download("Vision-Decision/glance-train-data", f"arch/phase2/{f}", repo_type="dataset", local_dir="/content/gtd")
        shutil.copy(p, f"/content/data/prep_arch/{f}"); print("[st2] got", f)
    except Exception as e:
        print("[st2] missing", f, repr(e)[:100])
miss = sum(1 for l in open("/content/data/prep_arch/train_base_avail.jsonl")
           if (r := json.loads(l)).get("img") and not os.path.exists(f"/content/data/{r['img']}"))
print("[st2] missing images in mixture:", miss)
PY
echo "[st2] DONE $(date -u +%T)"
