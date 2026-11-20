#!/usr/bin/env bash
# Robustness sprint: VM setup (after scripts/sync.sh SESSION, HF token + data/prep_v1 natural pools uploaded).
#   MODE=full : env + Qwen3.5-4B + COCO train2014 + v1 ckpt + v1 replay pools (text/gui/pixmo) + v1 mixture
#   MODE=eval : env + weights + COCO + v1 ckpt + GUI eval sets only (no training pools)
# Usage (VM): MODE=full bash /content/dv/scripts/sprint_robust/vm_setup.sh > /content/sr_setup.log 2>&1   (idempotent)
MODE=${MODE:-full}
set -uo pipefail
cd /content/dv
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
echo "[sr] $(date) start MODE=$MODE"
[ -f /content/evals.tgz ] && tar xzf /content/evals.tgz -C /content/dv   # evals/ is not part of scripts/sync.sh
bash scripts/colab_setup.sh > /content/colab_setup.log 2>&1; echo "[sr] colab_setup rc=$? $(tail -1 /content/colab_setup.log)"
python - <<'EOF'
from huggingface_hub import hf_hub_download
for f in ["v1/ckpt_final.pt"]:
    print(hf_hub_download("Vision-Decision/glance-train-ckpts-legacy", f, local_dir="/content/ckpt"))
EOF
if [ "$MODE" = "eval" ]; then
  mkdir -p /content/data/text
  python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("jgeuter/OpenDecisions", repo_type="dataset", local_dir="/content/data/text/opendecisions")
EOF
  python evals/gui_data.py --root /content/gui > /content/gui_data.log 2>&1; echo "[sr] gui_data rc=$?"
  echo "[sr] $(date) DONE"; exit 0
fi
bash scripts/v1/fetch.sh > /content/fetch.log 2>&1; echo "[sr] fetch rc=$?"
O=/content/data/prep_v1; mkdir -p $O
cp -n /content/up/pool_natural.jsonl /content/up/calib_natural.jsonl $O/ 2>/dev/null
bash scripts/v1/prep_all.sh > /content/prep_all.log 2>&1; echo "[sr] prep_all rc=$?"
python -m dv.data.prep_v1 postdedup --out $O > $O/log_postdedup.txt 2>&1; echo "[sr] postdedup rc=$?"
python -m dv.data.prep_v1 mix --out $O --total 1000000 > $O/log_mix.txt 2>&1; echo "[sr] mix rc=$?"
wc -l $O/*.jsonl
echo "[sr] $(date) DONE"; df -h /content | tail -1
