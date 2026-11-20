#!/usr/bin/env bash
# Final: the chosen recipe applied to v1.1 (Vision-Decision/glance-train-ckpts-legacy/v1.1/ckpt_final.pt), and v1.1 itself evaluated
# on the same sets. Recipe = /content/up/final_recipe.env (exports VARIANTS, TRAIN_SET, optional MIX_ARGS / STEPS).
cd /content/dv; export PYTHONPATH=/content/dv
until python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download("Vision-Decision/glance-train-ckpts-legacy", "v1.1/ckpt_final.pt", local_dir="/content/ckpt")
EOF
do sleep 300; done
until [ -s /content/up/final_recipe.env ]; do sleep 60; done
while pgrep -f "[b]ase_gen.py|[e]val.py|[t]rain.py" > /dev/null; do sleep 30; done
NAME=e_v11 EVAL_ONLY=1 INIT=/content/ckpt/v1.1/ckpt_final.pt bash scripts/sprint_robust/run_recipe.sh > /content/sr/e_v11.log 2>&1
set -a; . /content/up/final_recipe.env; set +a
NAME=f_v11 INIT=/content/ckpt/v1.1/ckpt_final.pt CFG=configs/v1.1_4b.json bash scripts/sprint_robust/run_recipe.sh > /content/sr/f_v11_run.log 2>&1
echo "[f_v11] DONE $(date -u)"
