#!/usr/bin/env bash
# Cheap retraining options on a VM: continue v1 on the hard-negative mixture (guisyn verified negatives + natural replay),
# optionally with the proper-scoring kappa variant; dump rows; push ckpt + rows to the private HF repo.
# Usage (VM): RUN=hnft STEPS=600 EXTRA="w_iou_brier=1.0 iou_head_on_sel=true" bash vm_train.sh > /content/sabs/train_$RUN.log 2>&1
set -u
RUN=${RUN:-hnft}; STEPS=${STEPS:-600}; EXTRA=${EXTRA:-}; REPO=Vision-Decision/glance-train-ckpts-legacy
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/content/sabs; mkdir -p $S/data
cd /content/dv
echo "[train] $(date -u) setup"
[ -f /content/data/coco/train2014/.done ] || bash scripts/colab_setup.sh > $S/setup.log 2>&1
mkdir -p /content/ckpt/v1
[ -f /content/ckpt/v1/ckpt.pt ] || { hf download $REPO v1/ckpt_final.pt --repo-type model --local-dir /content/ckpt/dl_v1 >/dev/null && mv /content/ckpt/dl_v1/v1/ckpt_final.pt /content/ckpt/v1/ckpt.pt; }
[ -d /content/guisyn/shards ] || hf download Vision-Decision/guisyn-v0 --repo-type dataset --local-dir /content/guisyn --include "shards/*" --include "stats.json" >/dev/null
[ -s $S/data/hnft.jsonl ] || python scripts/sprint_abs/make_hnft.py --pool_natural $S/data/pool_natural.jsonl --guisyn_shards /content/guisyn \
   --calib $S/data/guisyn_calib.jsonl --out $S/data/hnft.jsonl
# calibration images for later dumps
[ -f $S/data/.guisyn_calib_extracted ] || { python - <<'PY'
import json, tarfile, glob, collections
need = collections.defaultdict(set)
names = {json.loads(l)["img"][len("guisyn/"):] for l in open("/content/sabs/data/guisyn_calib.jsonl")}
for jf in glob.glob("/content/guisyn/shards/*/*.images.tar"):
    with tarfile.open(jf) as t:
        t.extractall("/content/data/guisyn", members=[m for m in t.getmembers() if m.name in names], filter="data")
PY
touch $S/data/.guisyn_calib_extracted; }
echo "[train] $(date -u) train $RUN"
O=/content/out/$RUN
[ -f $O/ckpt_final.pt ] || python train.py --cfg configs/v1_4b.json --set init_ckpt=/content/ckpt/v1/ckpt.pt train_file=$S/data/hnft.jsonl \
   max_steps=$STEPS lr=5e-5 lr_heads=1e-4 warmup_steps=30 out_dir=$O save_every_min=20 $EXTRA
ls -la $O
timeout 1800 hf upload $REPO $O/ckpt_final.pt sprint_abs/$RUN/ckpt_final.pt --repo-type model --commit-message "sprint_abs $RUN" >/dev/null 2>&1 && echo pushed ckpt
timeout 600 hf upload $REPO $O/train_log.jsonl sprint_abs/$RUN/train_log.jsonl --repo-type model >/dev/null 2>&1
echo "[train] $(date -u) dump"
[ -f /content/gui/venus.jsonl ] || python evals/gui_data.py --root /content/gui --only ss2,venus > $S/gui_data.log 2>&1
python scripts/sprint_abs/dump.py --ckpt $O/ckpt_final.pt --out $S/rows_$RUN --calib_dir $S/data --prep_dir /content/dv/data/prep \
  --sets ${DSETS:-calib_natural,grefcalib,guisyn,refcoco_val,refcocog_test,grec_val,grec_testA,grec_testB,ss2,venus}
timeout 1800 hf upload $REPO $S/rows_$RUN sprint_abs/rows_$RUN --repo-type model --commit-message "sprint_abs rows $RUN" >/dev/null 2>&1 && echo pushed rows
echo "[train] $(date -u) DONE"
