#!/usr/bin/env bash
# gx-b: arm B seed 0 (background) + equivalence on the released checkpoint + released eval in all three directions.
until grep -q "gx-setup\] DONE" /content/gx/setup.log 2>/dev/null; do sleep 30; done
cd /content/dv
RUN=i2t_only_s0 SEED=0 nohup bash scripts/glancex/vm_train_gx.sh > /content/gx/train_i2t_only_s0.log 2>&1 &
bash scripts/glancex/vm_equivalence.sh > /content/gx/equiv.log 2>&1
SRC=Vision-Decision/Glance-X-4B TAG=released MODES="image_first text_first joint" bash scripts/glancex/vm_eval.sh > /content/gx/eval_released.log 2>&1
echo CHAIN_B_DONE
