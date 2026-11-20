#!/usr/bin/env bash
# v1.2 post-training evaluation on the VM (PREREG groups A, B, C; group D uses the S2 sprint pipeline, see README).
# Usage (VM): nohup bash /content/dv/scripts/v12/vm_post.sh [CKPT] > /content/v12_post.log 2>&1 &
set -uo pipefail
C=${1:-/content/out/v1_2/ckpt_final.pt}; O=/content/out/v1_2/post; REPO=Vision-Decision/glance-train-ckpts
mkdir -p $O; cd /content/dv
export PYTHONPATH=/content/dv HF_HUB_OFFLINE=1
until [ -f $C ]; do sleep 60; done
echo "[post] start $(date -u)"
# B: Glance-Data diagnostic (+ pos_pos_sole add-on)
python scripts/ground_v2/eval_diag.py --ckpt $C --recs /content/gd/diagnostic.jsonl,/content/gd/diagnostic_pos_sole.jsonl \
  --out $O/diag.json --rows $O/diag_rows.jsonl.gz; echo "[post] diag done $(date -u)"
# A: glance-robust-ref
python eval.py --ckpt $C --calib_dir '' --prep_dir /content/robust --sets robust_zh,robust_para,robust_func,robust_vague \
  --rows_dir $O/robust_rows --out $O/robust_eval.json --num_workers 4
python scripts/robust/score.py --prep /content/robust --rows v1.2=$O/robust_rows --out $O/robust_scores.json
echo "[post] robust done $(date -u)"
# C: natural + text (temperatures from calib splits), then GUI at native 8192 tokens with few workers (/dev/shm)
python eval.py --ckpt $C --calib_dir /content/data/prep_v1 --latency --out $O/eval_results.json --rows_dir $O/rows --num_workers 4 \
  --sets refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test,grefcoco_val_merged,grec_val,grec_testA,grec_testB,opendecisions
echo "[post] natural/text done $(date -u)"
python eval.py --ckpt $C --calib_dir /content/data/prep_v1 --out $O/eval_results_gui.json --rows_dir $O/rows_gui --num_workers 2 \
  --gui_max_vis 8192 --gui_upscale 1.0 --zoom sspro --zoom_frac 0.4 --zoom_max_vis 4096 --sets ss2,venus,sspro
echo "[post] gui done $(date -u)"
unset HF_HUB_OFFLINE
timeout 3600 hf upload $REPO $O v1.2/post --repo-type model --commit-message "v1.2 post-training eval" > /dev/null && echo "[post] pushed"
echo "[post] ALL DONE $(date -u)"
