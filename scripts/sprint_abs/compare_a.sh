#!/usr/bin/env bash
# sabs-a (has the v1 rows): fetch the retrained variants' rows from the private repo and compare each with v1.
cd /content/dv
export PYTHONPATH=/content/dv
S=/content/sabs; REPO=Vision-Decision/glance-train-ckpts-legacy
mkdir -p $S/cmp/v1
for f in $S/rows_v1/*.jsonl.gz $S/rows_v1p/*.jsonl.gz; do ln -sf $f $S/cmp/v1/; done
for RUN in hnft hnft_mt; do
  hf download $REPO --repo-type model --include "sprint_abs/rows_$RUN/*" --local-dir /content/dl_$RUN >/dev/null
  python scripts/sprint_abs/compare.py --a $S/cmp/v1 --b /content/dl_$RUN/sprint_abs/rows_$RUN --names v1,$RUN --boot 500 \
    --out $S/cmp/compare_${RUN}_v1.json --sets calib_natural,grefcalib,guisyn,refcoco_val,refcocog_test,grec_val,grec_testA,grec_testB,ss2,venus \
    > $S/cmp/compare_$RUN.log 2>&1
done
echo "[compare_a] DONE $(date -u)" > $S/cmp/done
