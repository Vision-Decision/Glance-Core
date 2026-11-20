#!/usr/bin/env bash
# Sprint (absence/confidence) VM pipeline: env + data, extended row dumps, official VenusBench-GD scorer, push rows to HF.
# Usage (VM): bash /content/dv/scripts/sprint_abs/vm_run.sh TAG CKPT_REPO_PATH  > /content/sabs/run_TAG.log 2>&1
#   TAG=v1  CKPT=v1/ckpt_final.pt   (private repo Vision-Decision/glance-train-ckpts)
set -u
TAG=${1:-v1}; CK=${2:-v1/ckpt_final.pt}; REPO=Vision-Decision/glance-train-ckpts
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/content/sabs; mkdir -p $S/data $S/rows_$TAG $S/venus_$TAG
cd /content/dv
echo "[sabs] $(date -u) setup"
[ -f /content/data/coco/train2014/.done ] || bash scripts/colab_setup.sh > $S/setup.log 2>&1
tail -2 $S/setup.log
# checkpoint
mkdir -p /content/ckpt/$TAG
[ -f /content/ckpt/$TAG/ckpt.pt ] || { hf download $REPO $CK --repo-type model --local-dir /content/ckpt/dl_$TAG >/dev/null && mv /content/ckpt/dl_$TAG/$CK /content/ckpt/$TAG/ckpt.pt; }
ls -la /content/ckpt/$TAG
# GUI eval sets
[ -f /content/gui/venus.jsonl ] || python evals/gui_data.py --root /content/gui --only ss2,venus > $S/gui_data.log 2>&1
tail -3 $S/gui_data.log
# guisyn (held-out GUI calibration source)
if [ ! -s $S/data/guisyn_calib.jsonl ]; then
  hf download Vision-Decision/guisyn-v0 --repo-type dataset --local-dir /content/guisyn --include "shards/*" --include "stats.json" >/dev/null
  python scripts/sprint_abs/guisyn_calib.py --shards /content/guisyn --out $S/data/guisyn_calib.jsonl
fi
# official VenusBench-GD evaluator
if [ ! -d /content/uivenus ]; then
  git clone -q -b VenusBench-GD https://github.com/inclusionAI/UI-Venus /content/uivenus
  cp scripts/sprint_abs/venus_glance.py /content/uivenus/models/glance.py
  python - <<'PY'
p = "/content/uivenus/models/__init__.py"; s = open(p).read()
if '"glance"' not in s:
    s = s.replace("MODEL_MAP = {", 'MODEL_MAP = {\n    "glance": ("models.glance", "GlanceModel"),', 1); open(p, "w").write(s)
PY
fi
echo "[sabs] $(date -u) dump"
python scripts/sprint_abs/dump.py --ckpt /content/ckpt/$TAG/ckpt.pt --out $S/rows_$TAG --calib_dir $S/data --prep_dir /content/dv/data/prep \
  --sets ${SETS:-calib_natural,grefcalib,guisyn,refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test,grec_val,grec_testA,grec_testB,ss2,venus} \
  ${DUMP_ARGS:-}
echo "[sabs] $(date -u) venus forward cache (official scoring: chain_venus.sh / venus_official.sh)"
VR=/content/gui/venus_raw
[ -f $S/venus_$TAG/cache.json ] || python scripts/sprint_abs/venus_cache_from_rows.py $S/rows_$TAG/venus.jsonl.gz $VR/instruction $VR/images $S/venus_$TAG/cache.json
python scripts/sprint_abs/fit_gui_cal.py $S/rows_$TAG/guisyn.jsonl.gz $S/data/platt_gui_$TAG.json $S/data/stack_gui_$TAG.json
ls -la $S/venus_$TAG
echo "[sabs] $(date -u) push"
timeout 1800 hf upload $REPO $S/rows_$TAG sprint_abs/rows_$TAG --repo-type model --commit-message "sprint_abs rows $TAG" >/dev/null 2>&1 && echo pushed rows
echo "[sabs] $(date -u) DONE"
