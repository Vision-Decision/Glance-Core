#!/usr/bin/env bash
# sg-train VM: env + ckpts + GUI benches, then training data fetch + GUI-only mixture.
cd /content/dv
bash evals/sprint/setup_vm.sh v1/ckpt_final.pt v1.1/ckpt_last.pt > /content/setup.log 2>&1
bash evals/sprint/fetch_train.sh > /content/fetch_train.log 2>&1
python evals/sprint/prep_train.py --out /content/data/prep_gui > /content/prep_train.log 2>&1
echo "[chain] DONE $(date -u +%T)"
