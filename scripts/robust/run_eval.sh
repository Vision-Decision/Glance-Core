#!/usr/bin/env bash
# VM: evaluate models on glance-robust-ref; per-row outputs + scores go to the private HF repo (not the laptop).
# Usage: bash run_eval.sh MODEL CKPT|base [presence_T]   e.g.  run_eval.sh v1 /content/ckpt/v1/ckpt_final.pt
M=$1; C=$2
cd /content/dv && export HF_HUB_OFFLINE=1
SETS=robust_zh,robust_para,robust_func,robust_vague
echo "[robust] $M start $(date -u)"
if [ "$C" = base ]; then
  python scripts/robust/base_gen.py --prep /content/robust --out_dir /content/rrows/$M
else
  python eval.py --ckpt $C --calib_dir '' --prep_dir /content/robust --sets $SETS --rows_dir /content/rrows/$M --out /content/rrows/$M/eval.json
fi
unset HF_HUB_OFFLINE
timeout 1800 hf upload Vision-Decision/glance-train-ckpts-legacy /content/rrows/$M robust_eval/$M --repo-type model --commit-message "robust-ref rows $M" > /dev/null 2>&1 && echo "[robust] $M pushed"
echo "[robust] $M done $(date -u)"
