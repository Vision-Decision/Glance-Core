#!/usr/bin/env bash
# gx-c: arm C (dv recipe on the Glance-X mix) and arm B seed 1, concurrently on one 96 GB GPU.
until grep -q "gx-setup\] DONE" /content/gx/setup.log 2>/dev/null; do sleep 30; done
cd /content/dv
RUN=dvmix_s0 SEED=0 nohup bash scripts/glancex/vm_train_dv.sh > /content/gx/train_dvmix_s0.log 2>&1 &
sleep 120
RUN=i2t_only_s1 SEED=1 bash scripts/glancex/vm_train_gx.sh > /content/gx/train_i2t_only_s1.log 2>&1
wait
echo CHAIN_C_DONE
