#!/usr/bin/env bash
# After the main queue: grammar-constrained eval of every AR run, then latency of every run (GPU otherwise idle).
cd /content/dv
until grep -q "ALL DONE" /content/out/queue.log; do sleep 60; done
for R in ar_digits_neg25 ar_atomic1000_neg25 ar_hier_neg25 ar_digits_neg05 ar_atomic100_neg25; do
  O=/content/out/abl/$R
  [ -f $O/ckpt_final.pt ] || continue
  [ -f $O/rows_c.json ] && continue
  echo "[post] eval_c $R $(date)"
  python -m ablation.eval_ab --ckpt $O/ckpt_final.pt --constrained --out $O/eval_c.json --rows $O/rows_c.json > $O/eval_c_stdout.log 2>&1 || echo "[post] EVAL_C FAIL $R"
done
for R in ours_neg25 ar_digits_neg25 ar_hier_neg25 ar_atomic1000_neg25 ar_atomic100_neg25 ours_neg05 ar_digits_neg05; do
  O=/content/out/abl/$R
  [ -f $O/ckpt_final.pt ] || continue
  [ -f $O/latency.json ] && continue
  echo "[post] latency $R $(date)"
  python -m ablation.eval_ab --ckpt $O/ckpt_final.pt --no_eval --latency --out $O/latency.json > $O/latency_stdout.log 2>&1 || echo "[post] LAT FAIL $R"
done
echo "[post] ALL DONE $(date)"
