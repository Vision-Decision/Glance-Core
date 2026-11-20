#!/usr/bin/env bash
# After v1.1 training: full eval (+temperatures incl. difficulty-matched choice T, high-res GUI, crop-zoom on SS-Pro,
# official VenusBench scorer, GREC on val/testA/testB, latency, per-row dumps), base on OpenDecisions, release export,
# and push of rows + release weights to the private HF repo (nothing bulky goes to the laptop).
OUT=${1:-/content/out/v1_1}; REPO=${2:-Vision-Decision/glance-train-ckpts-legacy}
cd /content/dv
until [ -f $OUT/ckpt_final.pt ]; do sleep 30; done
sleep 20
export HF_HUB_OFFLINE=1
echo "[post] eval start $(date)"
python eval.py --ckpt $OUT/ckpt_final.pt --calib_dir /content/data/prep_v1 --latency --out $OUT/eval_results.json \
  --rows_dir $OUT/rows --gui_max_vis 8192 --gui_upscale 1.0 --zoom sspro --zoom_frac 0.4 --zoom_max_vis 4096 \
  --sets refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test,grefcoco_val_merged,grec_val,grec_testA,grec_testB,opendecisions,ss2,sspro,venus
echo "[post] base start $(date)"
python eval.py --base --calib_dir /content/data/prep_v1 --out $OUT/eval_base_qwen35_4b.json --rows_dir $OUT/rows_base --sets opendecisions
echo "[post] export start $(date)"
python scripts/v1/export_release.py --ckpt $OUT/ckpt_final.pt --eval_json $OUT/eval_results.json --out /content/release/v1.1 --merged
unset HF_HUB_OFFLINE
echo "[post] push start $(date)"
timeout 3600 hf upload $REPO $OUT/rows v1.1/eval_rows --repo-type model --commit-message "v1.1 per-row eval outputs" 
timeout 3600 hf upload $REPO $OUT/rows_base v1.1/eval_rows_base --repo-type model
timeout 7200 hf upload $REPO /content/release/v1.1 v1.1/release --repo-type model --commit-message "v1.1 release weights (adapter, heads, merged bf16)"
echo "[post] ALL DONE $(date)"
