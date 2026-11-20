#!/usr/bin/env bash
# queue q_c (third VM): jobs moved off sr-a / sr-b
until [ -s /content/data/prep_v1/train.jsonl ] && grep -q wrote /content/data/prep_v1/log_mix.txt 2>/dev/null; do sleep 60; done
until [ -s /content/up/teacher_iou.json ]; do sleep 60; done
for j in r2_aug d_heads r5_teacher; do bash /content/dv/scripts/sprint_robust/jobs/$j.sh > /content/sr/$j.log 2>&1; done
echo "[q_c] ALL DONE $(date -u)"
