#!/usr/bin/env bash
# Phase-2 training VM: v1.1 raw data (fetch.sh + fetch_v11.sh), the v1.1-replay mixture train_base.jsonl from
# Vision-Decision/glance-train-data:v1.2/, filtered to records whose image exists. Usage (VM, detached):
#   bash /content/dv/arch/vm/setup_train.sh > /content/logs/setup_train.log 2>&1
set -uo pipefail
cd /content/dv
bash arch/vm/setup.sh "coco gui" "Vision-Decision/glance-train-ckpts-legacy:v1.1/ckpt_final.pt"
bash scripts/v1/fetch.sh > /content/logs/fetch.log 2>&1; echo "[st] fetch rc=$? $(date -u +%T)"
bash scripts/v1/fetch_v11.sh > /content/logs/fetch11.log 2>&1; echo "[st] fetch11 rc=$? $(date -u +%T)"
python - <<'PY'
import json, os
from huggingface_hub import hf_hub_download
p = hf_hub_download("Vision-Decision/glance-train-data", "v1.2/train_base.jsonl", repo_type="dataset", local_dir="/content/gtd")
os.makedirs("/content/data/prep_arch", exist_ok=True)
n = k = 0; miss = {}
with open("/content/data/prep_arch/train_base_avail.jsonl", "w") as f:
    for l in open(p):
        r = json.loads(l); n += 1
        if r.get("img") and not os.path.exists(f"/content/data/{r['img']}"):
            s = r["img"].split("/")[0] + "/" + r["img"].split("/")[1]; miss[s] = miss.get(s, 0) + 1
            continue
        f.write(l); k += 1
print("[st] train_base records", n, "kept", k, "missing by source", miss, flush=True)
PY
echo "[st] DONE $(date -u +%T)"; df -h /content | tail -1
