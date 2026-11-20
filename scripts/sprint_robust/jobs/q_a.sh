#!/usr/bin/env bash
# queue q_a: strictly serial GPU jobs; waits for the replay mixture and an idle GPU first
until [ -s /content/data/prep_v1/train.jsonl ] && grep -q wrote /content/data/prep_v1/log_mix.txt 2>/dev/null; do sleep 60; done
while pgrep -f '[b]ase_gen.py|[e]val.py|[t]rain.py' > /dev/null; do sleep 30; done
bash /content/dv/scripts/sprint_robust/jobs/e_rob_v1.sh > /content/sr/e_rob_v1.log 2>&1
bash /content/dv/scripts/sprint_robust/jobs/c0_ctrl.sh > /content/sr/c0_ctrl.log 2>&1
bash /content/dv/scripts/sprint_robust/jobs/r3_cons.sh > /content/sr/r3_cons.log 2>&1
bash /content/dv/scripts/sprint_robust/jobs/r4_lm.sh > /content/sr/r4_lm.log 2>&1
bash /content/dv/scripts/sprint_robust/jobs/r2_aug.sh > /content/sr/r2_aug.log 2>&1
echo "[q_a] ALL DONE $(date -u)"
