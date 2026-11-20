#!/usr/bin/env bash
# Phase-2 run: train one variant, evaluate (RefCOCO/gRefCOCO + ScreenSpot-Pro native), push ckpt + rows to the org HF repos.
#   bash arch/train/chain.sh NAME ["key=value ..."]   (config: configs/arch/NAME.json)
set -uo pipefail
N=$1; SETS=${2:-}
cd /content/dv; O=/content/out/arch/$N; mkdir -p $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[chain] $N start $(date -u +%T)"
bash arch/vm/ckpt_pusher.sh $O $N > $O/pusher.log 2>&1 &
[ -f $O/ckpt_final.pt ] || python train.py --cfg configs/arch/$N.json $SETS > $O/train.log 2>&1
[ -f $O/ckpt_final.pt ] || { echo "[chain] $N TRAIN FAILED"; tail -5 $O/train.log; exit 1; }
python - <<PY
from huggingface_hub import HfApi
api = HfApi()
for f in ["ckpt_final.pt", "config.json", "train_log.jsonl"]:
    api.upload_file(path_or_fileobj="$O/" + f, path_in_repo="arch/$N/" + f, repo_id="Vision-Decision/glance-train-ckpts", repo_type="model")
print("[chain] pushed ckpt")
PY
python -m arch.alpha.run --ckpt $O/ckpt_final.pt --alphas 1 --what coco --out $O/eval > $O/eval_coco.log 2>&1
python -m arch.alpha.run --ckpt $O/ckpt_final.pt --alphas 1 --what sspro --out $O/eval > $O/eval_sspro.log 2>&1
grep -a "\[alpha\]" $O/eval_coco.log $O/eval_sspro.log
python arch/vm/push.py $O/eval arch/phase2/$N/eval
echo "[chain] $N DONE $(date -u +%T)"
