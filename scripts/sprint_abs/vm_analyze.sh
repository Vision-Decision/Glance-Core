#!/usr/bin/env bash
# Run the offline analysis on the VM (CPU) and keep only small JSON/markdown outputs.
# Usage (VM): TAG=v1 [PEAKS=rows_v1p] bash vm_analyze.sh
set -u
TAG=${TAG:-v1}; S=/content/sabs; A=$S/an_$TAG
rm -rf $A/rows; mkdir -p $A/rows
for f in $S/rows_$TAG/*.jsonl.gz; do ln -sf $f $A/rows/; done
if [ -n "${PEAKS:-}" ]; then for f in $S/$PEAKS/*.jsonl.gz; do ln -sf $f $A/rows/; done; fi
cd /content/dv
python scripts/sprint_abs/analyze.py --rows $A/rows --venus $S/venus_$TAG --tag $TAG --out $A --boot ${BOOT:-1000} --reps ${REPS:-200} > $A/analyze.log 2>&1
python scripts/sprint_abs/tables.py $A/results.json > $A/tables.md 2>> $A/analyze.log
echo "[vm_analyze] DONE $(date -u)" >> $A/analyze.log
