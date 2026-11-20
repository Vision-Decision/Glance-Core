#!/usr/bin/env bash
# Gen-v1 VM pipeline (one Colab G4). Idempotent; progress in /content/out/gen_v1.log.
#   STAGE=smoke bash gen_v1/vm_run.sh   env + 4-step training on a small synthetic mix (/content/smoke_train.jsonl:
#                                       box / point / absent / choice / noul / score, image + text states) + 20-image dump
#   STAGE=train bash gen_v1/vm_run.sh   full training (v1 config, one epoch), checkpoints pushed to HF every 20 min
#   STAGE=dump  bash gen_v1/vm_run.sh   read-outs on the eval sets, pushed to HF
set -u
cd /content/dv
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
REPO=Vision-Decision/glance-train-ckpts; P=baselines/gen_v1
O=/content/out/gen_v1; mkdir -p $O
LOG=/content/out/gen_v1.log
log() { echo "[gen_v1] $* $(date -u +%T)" >> $LOG; }
STAGE=${STAGE:-smoke}
log "stage $STAGE"
if [ ! -f /content/setup.done ]; then
  bash scripts/colab_setup.sh > /content/setup.log 2>&1 && touch /content/setup.done || { log SETUP FAIL; exit 1; }
  log setup ok
fi
if [ $STAGE != smoke ] && [ ! -f /content/data/prep_v1/train.jsonl ]; then
  mkdir -p /content/dl
  hf download $REPO $P/data/genv1_prep.tgz $P/data/genv1_images.tar --local-dir /content/dl > /dev/null 2>&1 || { log DATA DL FAIL; exit 1; }
  tar xzf /content/dl/$P/data/genv1_prep.tgz -C /content/data && tar xf /content/dl/$P/data/genv1_images.tar -C /content/data \
    && rm -rf /content/dl || { log DATA EXTRACT FAIL; exit 1; }
  log "data ok $(wc -l < /content/data/prep_v1/train.jsonl) records"
fi
if [ ! -f /content/gui/venus.jsonl ]; then
  (python evals/gui_data.py --root /content/gui --only ss2,venus > /content/gui_data.log 2>&1; log "gui data rc=$?") &
fi

if [ $STAGE = smoke ]; then
  S=/content/out/smoke; rm -rf $S
  python -m gen_v1.train --cfg configs/v1_4b.json --set out_dir=$S max_steps=4 log_every=1 train_file=/content/smoke_train.jsonl \
    > $S.train.log 2>&1 && log "smoke train ok" || { log SMOKE TRAIN FAIL; exit 1; }
  python -m gen_v1.dump --ckpt $S/ckpt_final.pt --out $S/rows --sets grefcoco_val_merged --limit 20 > $S.dump.log 2>&1 \
    && log "smoke dump ok" || { log SMOKE DUMP FAIL; exit 1; }
  log "smoke done"
fi

if [ $STAGE = train ]; then
  (bash scripts/v1/push_ckpt_hf.sh $O $REPO $P 1200 > /content/out/push.log 2>&1) &
  RES=""; [ -f $O/ckpt_last.pt ] && RES="resume=$O/ckpt_last.pt"
  python -m gen_v1.train --cfg configs/v1_4b.json --set out_dir=$O $RES >> $O/train_stdout.log 2>&1 \
    && log "train done" || { log TRAIN FAIL; exit 1; }
  wait
  timeout 600 hf upload $REPO $O/train_log.jsonl $P/train_log.jsonl --repo-type model > /dev/null 2>&1
  timeout 600 hf upload $REPO $O/train_stdout.log $P/train_stdout.log --repo-type model > /dev/null 2>&1
  log "train pushed"
fi

if [ $STAGE = dump ]; then
  while [ ! -f /content/gui/venus.jsonl ]; do sleep 20; done
  R=/content/out/gen_v1_rows
  python -m gen_v1.dump --ckpt $O/ckpt_final.pt --out $R \
    --sets ${SETS:-grefcoco_val_merged,arcal_gtest,arhead_rcg,ss2,venus,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test} \
    >> $R.log 2>&1 && log "dump done" || { log DUMP FAIL; }
  python -m gen_v1.iou_head --feat_dir $R/feats --out $R/head_scores.npz > $R/head_stdout.log 2>&1 && log "head done" || log "HEAD FAIL"
  timeout 1800 hf upload $REPO $R $P/rows --repo-type model --exclude "feats/*" --commit-message "gen_v1 rows" > /dev/null 2>&1 \
    && log "rows pushed" || log "ROWS PUSH FAIL"
  timeout 2400 hf upload $REPO $R/feats $P/feats --repo-type model --commit-message "gen_v1 feats" > /dev/null 2>&1 \
    && log "feats pushed" || log "FEATS PUSH FAIL"
fi
log "stage $STAGE finished"
