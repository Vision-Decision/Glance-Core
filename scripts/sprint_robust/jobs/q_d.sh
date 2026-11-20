#!/usr/bin/env bash
until [ -s /content/data/prep_v1/train.jsonl ] && grep -q wrote /content/data/prep_v1/log_mix.txt 2>/dev/null; do sleep 60; done
bash /content/dv/scripts/sprint_robust/jobs/f_v11.sh > /content/sr/f_v11.log 2>&1
