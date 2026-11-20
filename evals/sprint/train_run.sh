#!/usr/bin/env bash
# GUI-specialised continuation (sprint ft1) on sg-train.
#   train_run.sh INIT_HF_PATH RUN_NAME [extra --set args...]
# Builds the desktop-heavy GUI mixture (once), trains from INIT (LoRA + heads), and pushes checkpoints to
# Vision-Decision/glance-train-ckpts-legacy/sprint_gui/RUN_NAME while training.
INIT=${1:-v1.1/ckpt_last.pt}; RUN=${2:-ft1}; shift 2
cd /content/dv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/content/data/prep_gui_v2; mkdir -p $P
cp -n /content/data/prep_gui/pool_guisyn.jsonl $P/ 2>/dev/null
if [ ! -f $P/train.jsonl ]; then
  python evals/sprint/prep_train.py --out $P --total_q 400000 --max_repeat 1 \
    --img_cap web_guienv=1500,mobile_aw=1500,mobile_uibert=500,mobile_ricosca=200,mobile_widgetcap=200,guisyn=3000 \
    > /content/prep_v2.log 2>&1 || { echo "[train] prep FAIL"; exit 1; }
fi
echo "[train] prep done $(date -u +%T)"; tail -c 600 $P/stats.json
D=/content/ckpt/init_$RUN; mkdir -p $D
timeout 1800 hf download Vision-Decision/glance-train-ckpts-legacy $INIT --local-dir $D >/dev/null || { echo "[train] init download FAIL"; exit 1; }
OUT=/content/out/sprint_$RUN
nohup bash scripts/v1/push_ckpt_hf.sh $OUT Vision-Decision/glance-train-ckpts-legacy sprint_gui/$RUN 900 > /content/push_$RUN.log 2>&1 &
python train.py --cfg configs/sprint_gui_ft.json --set train_file=$P/train.jsonl out_dir=$OUT init_ckpt=$D/$INIT "$@" \
  > /content/train_$RUN.log 2>&1
echo "[train] $RUN rc=$? $(date -u +%T)"
