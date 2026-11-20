#!/usr/bin/env bash
# One robustness recipe on the VM: mixture -> fine-tune from INIT -> eval -> push LoRA+heads ckpt to HF.
# Usage (VM): NAME=r_zhpara VARIANTS=zh,para STEPS=800 TRAIN_SET="w_cons=1" MIX_ARGS="" \
#             bash /content/dv/scripts/sprint_robust/run_recipe.sh > /content/sr/NAME.log 2>&1
set -uo pipefail
cd /content/dv
export PYTHONPATH=/content/dv HF_HUB_OFFLINE=0 TQDM_DISABLE=1
NAME=${NAME:?}; VARIANTS=${VARIANTS:-none}; STEPS=${STEPS:-800}; TRAIN_SET=${TRAIN_SET:-}; MIX_ARGS=${MIX_ARGS:-}
INIT=${INIT:-/content/ckpt/v1/ckpt_final.pt}; LR=${LR:-1e-4}; LR_HEADS=${LR_HEADS:-$LR}; EVAL_ONLY=${EVAL_ONLY:-0}
SETS=${SETS:-rg,robust_zh,robust_para,robust_func,robust_vague,refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocog_val,grefcoco_val_merged,opendecisions,ss2,sspro,venus}
OUT=/content/out/sr/$NAME; mkdir -p $OUT
echo "[run] $NAME start $(date -u) variants=$VARIANTS steps=$STEPS set=$TRAIN_SET"
if [ "$EVAL_ONLY" = "0" ]; then
  python scripts/sprint_robust/mix_robust.py --variants $VARIANTS --out $OUT/mix.jsonl $MIX_ARGS > $OUT/mix_stats.json || exit 1
  python train.py --cfg ${CFG:-configs/v1_4b.json} --set train_file=$OUT/mix.jsonl out_dir=$OUT init_ckpt=$INIT max_steps=$STEPS \
     lr=$LR lr_heads=$LR_HEADS warmup_steps=50 save_every_min=20 $TRAIN_SET > $OUT/train.log 2>&1 || { echo "[run] train failed"; tail -30 $OUT/train.log; exit 1; }
  CK=$OUT/ckpt_final.pt
else
  CK=$INIT
fi
echo "[run] eval $(date -u)"
python eval.py --ckpt $CK --calib_dir /content/data/prep_v1 --prep_dir /content/dv/data/prep --sets $SETS \
   --rows_dir $OUT/rows --out $OUT/eval_results.json > $OUT/eval.log 2>&1 || echo "[run] eval rc=$?"
if [ "$EVAL_ONLY" = "0" ]; then
  python scripts/export_ckpt.py $CK $OUT/ckpt_final_bf16.pt > /dev/null 2>&1
  python - <<EOF
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="$OUT/ckpt_final_bf16.pt", path_in_repo="sprint_robust/$NAME/ckpt_final_bf16.pt",
                    repo_id="Vision-Decision/glance-train-ckpts-legacy", repo_type="model")
print("pushed")
EOF
fi
echo "[run] $NAME DONE $(date -u)"
