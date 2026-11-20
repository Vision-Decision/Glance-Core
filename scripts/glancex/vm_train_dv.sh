#!/usr/bin/env bash
# Missing control, arm C: our recipe (dv train.py, Qwen3.5-4B instruct, image-first shared prefix, dv box heads; the
# box settings of v0 / v1.1) on Glance-X's data mix (requests with <= 1 target; >= 34% original per image), 6,100
# steps of 8 images. Then eval.py rows on RefCOCO + gRefCOCO + the diagnostic set, pushed to HF.
# Usage (VM): RUN=dvmix_s0 SEED=0 bash /content/dv/scripts/glancex/vm_train_dv.sh > /content/gx/train_RUN.log 2>&1
set -uo pipefail
RUN=${RUN:-dvmix_s0}; SEED=${SEED:-0}; REPO=Vision-Decision/glance-train-ckpts
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/content/gx/train/$RUN; mkdir -p $O
push() { timeout 1800 hf upload $REPO $O glancex/train/$RUN --repo-type model --include "*.json" --include "*.jsonl" --include "*.log" --include "ckpt_*.pt" --include "opt_last.pt" \
           --commit-message "glancex train $RUN $1" >/dev/null 2>&1 && echo "[gx-train] pushed $1 $(date -u)"; }
( while sleep 1800; do [ -f $O/DONE ] && break; cp /content/gx/train_$RUN.log $O/train.log 2>/dev/null; push periodic; done ) &
RES=""; [ -f $O/ckpt_last.pt ] && RES="resume=$O/ckpt_last.pt"
cd /content/dv
python train.py --cfg configs/glancex_dvmix_4b.json --set seed=$SEED out_dir=$O $RES
touch $O/DONE; cp /content/gx/train_$RUN.log $O/train.log; push final
E=/content/gx/rows/$RUN/dv_eval; mkdir -p $E
python eval.py --ckpt $O/ckpt_final.pt --prep_dir /content/gxprep --out $E/eval_results.json --rows_dir $E \
  --sets refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test,grefcoco_val_merged,grefcoco_testA,grefcoco_testB,gxdiag
G=/content/gx/rows/$RUN/image_first; mkdir -p $G
python scripts/glancex/dv_rows_to_gx.py --rows $E --prep_dir /content/gxprep --out $G && python scripts/glancex/score.py $G $G/score.json | tail -1
timeout 1800 hf upload $REPO /content/gx/rows/$RUN results/glancex/$RUN --repo-type model --commit-message "glancex rows $RUN" >/dev/null 2>&1 && echo pushed rows
echo "[gx-train] DONE $(date -u)"
