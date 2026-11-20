#!/usr/bin/env bash
# v1 raw-data download on the Colab VM (after scripts/colab_setup.sh). Idempotent; each step logs and continues.
# Usage (VM): bash /content/dv/scripts/v1/fetch.sh > /content/fetch.log 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=0 PIP_DISABLE_PIP_VERSION_CHECK=1
R=/content/data; mkdir -p $R/text $R/gui $R/pixmo $R/evalimg
cd $R
echo "[fetch] $(date) start; disk:"; df -h /content | tail -1
pip install -q imagehash 2>/dev/null
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import os, traceback
R = "/content/data"
jobs = [
  ("tasksource/tasksource-jev-typed-decisions", f"{R}/text/tasksource", ["data/*", "sources.yaml"]),
  ("tasksource/procedural-typed-decisions", f"{R}/text/procedural", ["all/train-*", "all/validation-*", "README.md"]),
  ("jgeuter/DecisionBank", f"{R}/text/decisionbank", ["data/DecisionBank-260k/*", "README.md"]),
  ("inclusionAI/LemonDecider-Decision-Mix", f"{R}/text/lemon", ["train.jsonl", "row_licenses.jsonl", "upstream_aggregate.tsv", "validation.json"]),
  ("jgeuter/OpenDecisions", f"{R}/text/opendecisions", None),
  ("allenai/pixmo-points", f"{R}/pixmo/points", None),
  ("allenai/pixmo-count", f"{R}/pixmo/count", None),
  ("ServiceNow/GroundCUA", f"{R}/gui/groundcua", ["instruction_tuning.tar.gz", "README.md"]),
  ("OS-Copilot/OS-Atlas-data", f"{R}/gui/osatlas", ["desktop_domain/linux_*", "desktop_domain/macos_*", "mobile_domain/UIBert.zip",
       "mobile_domain/uibert_raw.json", "mobile_domain/mobile_images.zip", "mobile_domain/widget_captioning.json",
       "mobile_domain/ricosca.json", "mobile_domain/aw_mobile.json", "README.md"]),
  ("xlangai/aguvis-stage1", f"{R}/gui/aguvis", ["guienv.json", "guienvs.zip", "README.md"]),
  ("PointArena/pointarena-data", f"{R}/evalimg/pointarena", None),
]
for rid, d, pat in jobs:
    try:
        p = snapshot_download(rid, repo_type="dataset", local_dir=d, allow_patterns=pat, max_workers=16)
        print("[fetch] ok", rid, flush=True)
    except Exception as e:
        print("[fetch] FAIL", rid, repr(e)[:300], flush=True)
PY
echo "[fetch] $(date) hub done"; du -sh $R/* $R/gui/* $R/text/* 2>/dev/null
# unzip GUI images
cd $R/gui/osatlas
for z in desktop_domain/linux_images.zip desktop_domain/macos_images.zip mobile_domain/UIBert.zip mobile_domain/mobile_images.zip; do
  [ -f "$z.done" ] || { unzip -q -o "$z" -d "$(dirname $z)/img" && touch "$z.done" && rm -f "$z"; echo "[fetch] unzipped $z"; }
done
cd $R/gui/aguvis
[ -f guienvs.zip.done ] || { unzip -q -o guienvs.zip -d img && touch guienvs.zip.done && rm -f guienvs.zip; echo "[fetch] unzipped guienvs"; }
# eval sets for dedup / eval (other agent's script, used read-only)
cd /content/dv && python evals/gui_data.py --root /content/gui || echo "[fetch] gui_data FAIL"
echo "[fetch] $(date) DONE"; df -h /content | tail -1
