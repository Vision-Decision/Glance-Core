#!/usr/bin/env bash
# Background: every 40 min push OUT/ckpt_last.pt (+ train_log) to Vision-Decision/glance-train-ckpts:arch/NAME/ so a lost VM
# loses at most 40 min of training.   bash arch/vm/ckpt_pusher.sh OUT NAME &
O=$1; N=$2
while [ ! -f $O/ckpt_final.pt ]; do
  sleep 2400
  for f in ckpt_last.pt opt_last.pt train_log.jsonl; do
    [ -f $O/$f ] && python -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$O/$f', path_in_repo='arch/$N/$f', repo_id='Vision-Decision/glance-train-ckpts', repo_type='model')" >/dev/null 2>&1
  done
  echo "[pusher] $(date -u +%T) pushed $N"
done
