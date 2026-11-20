#!/usr/bin/env bash
# Throughput benchmark: 60 steps per variant alone, then ar_digits + ours concurrently.
cd /content/dv
B=/content/out/bench; mkdir -p $B
S="max_steps=60 log_every=20 save_every=100000"
for R in ours_neg25 ar_digits_neg25 ar_hier_neg25 ar_atomic1000_neg25; do
  python -m ablation.train_ab --cfg ablation/configs/$R.json --set $S out_dir=$B/$R > $B/$R.log 2>&1
  echo "[bench] alone $R $(grep '"step": 60' $B/$R.log | head -c 600)"
done
python -m ablation.train_ab --cfg ablation/configs/ar_digits_neg25.json --set $S out_dir=$B/par_a > $B/par_a.log 2>&1 &
python -m ablation.train_ab --cfg ablation/configs/ours_neg25.json --set $S out_dir=$B/par_b > $B/par_b.log 2>&1 &
wait
echo "[bench] par a $(grep '"step": 60' $B/par_a.log | head -c 600)"
echo "[bench] par b $(grep '"step": 60' $B/par_b.log | head -c 600)"
echo "[bench] DONE"
