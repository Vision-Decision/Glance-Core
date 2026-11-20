#!/usr/bin/env bash
# sabs-a: after the hard-negative run, evaluate v1.1 (private repo v1.1/ckpt_final.pt) with the same sprint pipeline.
# GUI sets use the v1 eval resolution (2048 visual tokens, upscale 2) because v1.1's native 8192-token GUI setting does
# not fit the A100-40GB vision tower; this is disclosed with the results.
cd /content/dv
export PYTHONPATH=/content/dv
S=/content/sabs
until grep -q "\[train\].*DONE" $S/train_hnft.log 2>/dev/null; do sleep 60; done
until python - <<'PY'
import sys
from huggingface_hub import HfApi
fs = HfApi().list_repo_files("Vision-Decision/glance-train-ckpts-legacy")
sys.exit(0 if "v1.1/ckpt_final.pt" in fs else 1)
PY
do sleep 300; done
bash scripts/sprint_abs/vm_run.sh v1.1 v1.1/ckpt_final.pt > $S/run_v1.1.log 2>&1
TAG=v1.1 bash scripts/sprint_abs/chain_venus.sh > $S/chain_venus_v11.log 2>&1
TAG=v1.1 bash scripts/sprint_abs/vm_analyze.sh
echo "[chain_v11] DONE $(date -u)" > $S/chain_v11.done
