#!/usr/bin/env bash
# v1.2 training on the VM (after build_mix.py). Pushes ckpt_last (+ optimizer) to HF every hour and ckpt_final at the end.
# Usage (VM): nohup bash /content/dv/scripts/v12/vm_train.sh [extra --set args] > /content/v12_train.log 2>&1 &
# Resume after a VM loss: download v1.2/ckpt_last.pt + opt_last.pt into /content/out/v1_2 and pass resume=/content/out/v1_2/ckpt_last.pt
set -uo pipefail
OUT=/content/out/v1_2; REPO=Vision-Decision/glance-train-ckpts
mkdir -p $OUT; cd /content/dv
export PYTHONPATH=/content/dv HF_HUB_OFFLINE=1
bash scripts/v12/vm_push.sh $OUT v1.2 > /content/v12_push.log 2>&1 &  # hourly ckpt_last, then ckpt_final
python train.py --cfg configs/v1.2_4b.json --set "$@"
echo "[train] exit rc=$? $(date -u)"
wait  # vm_push.sh uploads ckpt_final
