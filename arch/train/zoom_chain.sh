#!/usr/bin/env bash
# Phase-2 trained zoom: train (arch/train/train_zoom.py), push, evaluate first pass + in-block zoom (alliso) on SS-Pro native,
# and RefCOCO/gRefCOCO guards.   bash arch/train/zoom_chain.sh NAME "extra train args"
set -uo pipefail
N=$1; X=${2:-}; AL=${ALPHA:-1.0}
cd /content/dv; O=/content/out/arch/$N; mkdir -p $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK1=/content/ckpt/glance-train-ckpts/v1.1/ckpt_final.pt
bash arch/vm/ckpt_pusher.sh $O $N > $O/pusher.log 2>&1 &
[ -f $O/ckpt_final.pt ] || python -m arch.train.train_zoom --init $CK1 --train /content/data/prep_arch/train_base_avail.jsonl --out $O $X > $O/train.log 2>&1
[ -f $O/ckpt_final.pt ] || { echo "[zchain] TRAIN FAILED"; tail -5 $O/train.log; exit 1; }
python - <<PY
from huggingface_hub import HfApi
api = HfApi()
for f in ["ckpt_final.pt", "train_log.jsonl", "args.json"]:
    api.upload_file(path_or_fileobj="$O/" + f, path_in_repo="arch/$N/" + f, repo_id="Vision-Decision/glance-train-ckpts", repo_type="model")
print("[zchain] pushed")
PY
python -m arch.zoom.run_zoom --ckpt $O/ckpt_final.pt --prefix_vis 16384 --crop_vis 4096 --layout pre_alliso --alpha $AL --reprefill --out $O/eval/sspro_alliso > $O/eval_zoom.log 2>&1
python -m arch.zoom.run_zoom --ckpt $O/ckpt_final.pt --prefix_vis 2048 --crop_vis 4096 --layout pre_alliso --alpha $AL --out $O/eval/sspro_p2048_alliso > $O/eval_zoom2.log 2>&1
python -m arch.alpha.run --ckpt $O/ckpt_final.pt --alphas $AL --what coco --out $O/eval > $O/eval_coco.log 2>&1
grep -a "DONE" $O/eval_zoom*.log | grep -o '"acc1": [0-9.]*\|"acc2": [0-9.]*\|"acc_rp": [0-9.]*'; grep -a "\[alpha\]" $O/eval_coco.log
python arch/vm/push.py $O/eval arch/phase2/$N/eval
echo "[zchain] $N DONE $(date -u +%T)"
