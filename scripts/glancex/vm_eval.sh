#!/usr/bin/env bash
# Rows for a Glance-X checkpoint in several directions, pushed to HF after each direction.
# Usage (VM): SRC=Vision-Decision/Glance-X-4B TAG=released MODES="image_first text_first joint" IMPL=glancex \
#             bash /content/dv/scripts/glancex/vm_eval.sh > /content/gx/eval_TAG.log 2>&1
set -uo pipefail
SRC=${SRC:-Vision-Decision/Glance-X-4B}; TAG=${TAG:-released}; MODES=${MODES:-"image_first text_first joint"}; IMPL=${IMPL:-glancex}
SETS=${SETS:-refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test,grefcoco_val_merged,grefcoco_testA,grefcoco_testB,diag}
REPO=Vision-Decision/glance-train-ckpts
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /content/dv
for MODE in $MODES; do
  OUT=/content/gx/rows/$TAG/$MODE
  echo "[gx-eval] $(date -u) $TAG $MODE"
  python scripts/glancex/dump_rows.py --src "$SRC" --mode $MODE --impl $IMPL --sets $SETS --prep_dir /content/dv/data/prep \
     --diag /content/gxdata/diagnostic.jsonl --out $OUT ${DUMP_ARGS:-}
  python scripts/glancex/score.py $OUT $OUT/score.json | tail -1
  timeout 1800 hf upload $REPO $OUT results/glancex/$TAG/$MODE --repo-type model --exclude "*.done" \
     --commit-message "glancex rows $TAG $MODE" >/dev/null 2>&1 && echo "[gx-eval] pushed $TAG/$MODE"
done
echo "[gx-eval] DONE $(date -u)"
