#!/usr/bin/env bash
# After the hard-negative run: hard negatives + multi-target (one question per referred object) fine-tune from v1.
cd /content/dv
S=/content/sabs
until grep -q "\[train\].*DONE" $S/train_hnft.log 2>/dev/null; do sleep 30; done
python - <<'PY'
import random
rng = random.Random(3)
L = open("/content/sabs/data/hnft.jsonl").readlines() + open("/content/sabs/data/mt.jsonl").readlines()
rng.shuffle(L)
open("/content/sabs/data/hnft_mt.jsonl", "w").writelines(L)
print("hnft_mt records", len(L))
PY
RUN=hnft_mt STEPS=${STEPS:-300} TRAIN_FILE=$S/data/hnft_mt.jsonl bash scripts/sprint_abs/vm_train.sh > $S/train_hnft_mt.log 2>&1
echo "[chain_b] DONE $(date -u)" > $S/chain_b.done
