#!/usr/bin/env bash
# Fresh-VM chain: 27B env+weights, v0 env+COCO, then the full 27B run. Run detached on the VM
# after `scripts/sync.sh SESSION` and uploading scripts/27b/* to /content/27b/.
#   CMD="bash /content/27b/bootstrap_full.sh" LOG=/content/out/bootstrap_27b.log python3 scripts/remote_bg.py | colab-pool exec SESSION
# Extra args are passed to train_27b.sh full as key=value overrides.
set -euo pipefail
mkdir -p /content/out /content/27b_out
echo "[boot] $(date -u) 27b setup"
bash /content/27b/setup.sh > /content/27b_out/setup.log 2>&1
echo "[boot] $(date -u) v0 setup (COCO)"
pip uninstall -y -q torchao || true
bash /content/dv/scripts/colab_setup.sh > /content/out/setup_v0.log 2>&1
echo "[boot] $(date -u) launching full training -> /content/out/v0_27b.log"
bash /content/27b/train_27b.sh full "$@" > /content/out/v0_27b.log 2>&1
echo "[boot] $(date -u) training exited rc=$?"
