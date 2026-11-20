#!/usr/bin/env bash
# Analyse a retrained variant (RUN) once its dumps exist, and compare it with v1 on the same items (paired bootstrap).
RUN=${RUN:-hnft}; S=/content/sabs
cd /content/dv
until grep -q "\[train\].*DONE" $S/train_$RUN.log 2>/dev/null; do sleep 60; done
TAG=$RUN BOOT=${BOOT:-500} REPS=${REPS:-100} bash scripts/sprint_abs/vm_analyze.sh
mkdir -p $S/an_$RUN/v1ref; for f in $S/rows_v1/*.jsonl.gz $S/rows_v1p/*.jsonl.gz; do ln -sf $f $S/an_$RUN/v1ref/; done
python scripts/sprint_abs/compare.py --a $S/an_$RUN/v1ref --b $S/rows_$RUN --names v1,$RUN --out $S/an_$RUN/compare_v1.json \
  --sets calib_natural,grefcalib,guisyn,refcoco_val,refcocog_test,grec_val,grec_testA,grec_testB,ss2,venus > $S/an_$RUN/compare.log 2>&1
echo "[an_retrain] DONE $(date -u)" >> $S/an_$RUN/compare.log
