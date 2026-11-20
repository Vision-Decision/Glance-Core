#!/usr/bin/env bash
# ablation_v2 (VM side): train the given runs one after another (concurrent runs on one G4 measured 6.2 s/step each
# for 3 jobs vs ~1.5 s/step alone, i.e. slower in aggregate), evaluate each
# (ours: eval_ab; ar_*: grammar-constrained eval_ab, whose rows carry every field the analysis needs),
# then push the bf16 checkpoint to the private HF repo (ablation/<run>_bf16.pt).
# Idempotent: finished stages are skipped.   bash ablation/v2_queue.sh run1 run2 ...   (cwd /content/dv)
cd /content/dv
export PYTHONUNBUFFERED=1
LOG=/content/out/v2_queue.log
# optional one-shot pre-hook (e.g. a GPU benchmark that must run on an idle GPU before training starts)
[ -f /content/out/pre_queue.sh ] && { bash /content/out/pre_queue.sh; mv /content/out/pre_queue.sh /content/out/pre_queue.done; }
one() {
  R=$1; O=/content/out/abl/$R; mkdir -p $O
  if [ ! -f $O/ckpt_final.pt ]; then
    echo "[queue] train $R $(date)" >> $LOG
    python -m ablation.train_ab --cfg ablation/configs/$R.json > $O/train_stdout.log 2>&1 || { echo "[queue] TRAIN FAIL $R" >> $LOG; return; }
  fi
  case $R in ours*) EF=""; EV=eval;; *) EF="--constrained"; EV=eval_c;; esac
  if [ ! -f $O/rows_c.json ] && [ ! -f $O/rows.json ]; then
    echo "[queue] eval $R $(date)" >> $LOG
    python -m ablation.eval_ab --ckpt $O/ckpt_final.pt $EF --out $O/$EV.json --rows $O/rows_tmp.json > $O/${EV}_stdout.log 2>&1 \
      && { [ $EV = eval_c ] && mv $O/rows_tmp.json $O/rows_c.json || mv $O/rows_tmp.json $O/rows.json; } \
      || echo "[queue] EVAL FAIL $R" >> $LOG
  fi
  if [ ! -f $O/pushed ]; then
    python -m ablation.export_bf16 $O/ckpt_final.pt $O/ckpt_bf16.pt && \
    timeout 1800 hf upload Vision-Decision/glance-train-ckpts-legacy $O/ckpt_bf16.pt ablation/${R}_bf16.pt --repo-type model \
      --commit-message "ablation_v2 $R" > /dev/null 2>&1 && touch $O/pushed && rm -f $O/ckpt_bf16.pt
    [ -f $O/pushed ] || echo "[queue] PUSH FAIL $R" >> $LOG
  fi
  echo "[queue] done $R $(date)" >> $LOG
}
for R in "$@"; do one $R; done
echo "[queue] ALL DONE $* $(date)" >> $LOG
