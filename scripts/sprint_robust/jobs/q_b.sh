#!/usr/bin/env bash
# queue q_b: strictly serial GPU jobs; waits for the replay mixture and an idle GPU first
until [ -s /content/data/prep_v1/train.jsonl ] && grep -q wrote /content/data/prep_v1/log_mix.txt 2>/dev/null; do sleep 60; done
while pgrep -f '[b]ase_gen.py|[e]val.py|[t]rain.py' > /dev/null; do sleep 30; done
bash /content/dv/scripts/sprint_robust/jobs/r1_zhpara.sh > /content/sr/r1_zhpara.log 2>&1
bash /content/dv/scripts/sprint_robust/jobs/a_nozh.sh > /content/sr/a_nozh.log 2>&1
bash /content/dv/scripts/sprint_robust/jobs/a_nofunc.sh > /content/sr/a_nofunc.log 2>&1
bash /content/dv/scripts/sprint_robust/jobs/d_heads.sh > /content/sr/d_heads.log 2>&1
bash /content/dv/scripts/sprint_robust/jobs/r5_teacher.sh > /content/sr/r5_teacher.log 2>&1
echo "[q_b] ALL DONE $(date -u)"
