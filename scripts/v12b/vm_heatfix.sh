#!/usr/bin/env bash
# Fair head-off rerun of the heat arm (v1.2b heat loss diverged 1.3 -> 14 in the joint stage): same data, steps and schedule as
# vm_v12b.sh, plus own LR / grad clip for the heat map and the bounded focal normalisation (configs/v1.2b_heatfix_*.json).
# Runs on the v1.2 VM after the objectness arm and the zoom re-measure. Code: /content/dv_hf (glance-core).
# Usage (VM): nohup bash /content/dv_hf/scripts/v12b/vm_heatfix.sh > /content/heatfix.log 2>&1 &
set -uo pipefail
REPO=Vision-Decision/glance-train-ckpts; D=/content/dv_hf
cd $D; export PYTHONPATH=$D HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
until grep -q "\[zoom\] ALL DONE" /content/v12_zoom.log 2>/dev/null; do sleep 60; done
echo "[heatfix] start $(date -u)"
mkdir -p $D/data/prep && cp -n /content/dv_mb/data/prep/eval_*.jsonl $D/data/prep/
for ARM in heatfix; do
  P=v1.2b_heatfix
  for ST in heads joint; do
    CFG=configs/${P}_$ST.json; OUT=$(python -c "import json;print(json.load(open('$CFG'))['out_dir'])")
    python train.py --cfg $CFG --set limit_records=40 max_steps=3 warmup_steps=1 out_dir=/content/out/smoke_${ARM}_$ST save_every_min=0 \
      > /content/out/smoke_${ARM}_$ST.log 2>&1 || { echo "smoke $ARM $ST failed"; tail -30 /content/out/smoke_${ARM}_$ST.log; continue 2; }
    echo "[heatfix] train $ARM $ST $(date -u)"
    python train.py --cfg $CFG > /content/v12b/train_${ARM}_$ST.log 2>&1 || { echo "train $ARM $ST failed"; tail -30 /content/v12b/train_${ARM}_$ST.log; continue 2; }
    for f in ckpt_final.pt train_log.jsonl config.json; do
      timeout 1800 hf upload $REPO $OUT/$f $P/$ST/$f --repo-type model > /dev/null 2>&1 && echo "[heatfix] pushed $P/$ST/$f"
    done
  done
  C=$OUT/ckpt_final.pt; O=$OUT/post; mkdir -p $O
  echo "[heatfix] eval $ARM $(date -u)"
  python eval.py --ckpt $C --calib_dir /content/data/prep_v1 --prep_dir $D/data/prep --num_workers 4 --out $O/eval.json --rows_dir $O/rows \
    --sets refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test,grefcoco_val_merged,grec_val,cocoall_val,calibmb,diag \
    > $O/eval.log 2>&1 || tail -30 $O/eval.log
  python scripts/multibox/analyze_rows.py --calib $O/rows/calibmb.jsonl.gz --sets $O/rows/grec_val.jsonl.gz,$O/rows/cocoall_val.jsonl.gz,$O/rows/diag.jsonl.gz \
    > $O/analyze.txt 2>&1
  for M in thr count; do python scripts/v12b/multi_f1.py --rows $O/rows/grec_val.jsonl.gz --mode $M --calib $O/rows/calibmb.jsonl.gz > $O/multi_f1_$M.json 2>&1; done
  python scripts/ground_v2/eval_diag.py --summarize $O/rows/diag.jsonl.gz --out $O/diag_single_view.json > /dev/null 2>&1
  cp $O/eval.json $O/analyze.txt $O/multi_f1_*.json $O/diag_single_view.json $O/rows/ 2>/dev/null
  timeout 1800 hf upload $REPO $O/rows $P/post --repo-type model > /dev/null 2>&1 && echo "[heatfix] pushed $P eval"
done
echo "[heatfix] ALL DONE $(date -u)"
