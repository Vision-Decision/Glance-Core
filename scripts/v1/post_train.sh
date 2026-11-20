#!/usr/bin/env bash
# After v1 training: full eval (+temperatures, latency) and the zero-shot base on OpenDecisions.
OUT=${1:-/content/out/v1}
cd /content/dv
export HF_HUB_OFFLINE=1
until [ -f $OUT/ckpt_final.pt ]; do sleep 30; done
sleep 20
echo "[post] eval start $(date)"
python eval.py --ckpt $OUT/ckpt_final.pt --calib_dir /content/data/prep_v1 --latency --out $OUT/eval_results.json \
  --sets refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test,grefcoco_val_merged,opendecisions,ss2,sspro,venus
echo "[post] base start $(date)"
python eval.py --base --calib_dir /content/data/prep_v1 --out $OUT/eval_base_qwen35_4b.json --sets opendecisions
python scripts/export_ckpt.py $OUT/ckpt_final.pt $OUT/ckpt_final_bf16.pt
echo "[post] ALL DONE $(date)"
