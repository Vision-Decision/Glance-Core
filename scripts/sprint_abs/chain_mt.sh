#!/usr/bin/env bash
# sabs-b: hard negatives + multi-target fine-tune from v1 (independent VM).
cd /content/dv
export PYTHONPATH=/content/dv
S=/content/sabs; mkdir -p $S/data
[ -f /content/data/coco/train2014/.done ] || bash scripts/colab_setup.sh > $S/setup.log 2>&1
[ -d /content/guisyn/shards ] || hf download Vision-Decision/guisyn-v0 --repo-type dataset --local-dir /content/guisyn --include "shards/*" --include "stats.json" >/dev/null
[ -s $S/data/hnft.jsonl ] || python scripts/sprint_abs/make_hnft.py --pool_natural $S/data/pool_natural.jsonl --guisyn_shards /content/guisyn \
   --calib $S/data/guisyn_calib.jsonl --out $S/data/hnft.jsonl
python - <<'PY'
import random
rng = random.Random(3)
L = open("/content/sabs/data/hnft.jsonl").readlines() + open("/content/sabs/data/mt.jsonl").readlines()
rng.shuffle(L)
open("/content/sabs/data/hnft_mt.jsonl", "w").writelines(L)
print("hnft_mt records", len(L))
PY
[ -f /content/gui/venus.jsonl ] || python evals/gui_data.py --root /content/gui --only ss2,venus > $S/gui_data.log 2>&1
RUN=hnft_mt STEPS=${STEPS:-300} TRAIN_FILE=$S/data/hnft_mt.jsonl bash scripts/sprint_abs/vm_train.sh > $S/train_hnft_mt.log 2>&1
echo "[chain_mt] DONE $(date -u)" > $S/chain_mt.done
