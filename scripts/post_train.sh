#!/usr/bin/env bash
# Chain after training: wait for final ckpt -> full eval (+latency) -> generative baseline -> bf16 export.
OUT=${1:-/content/out/v0}
cd /content/dv
until [ -f $OUT/ckpt_final.pt ]; do sleep 30; done
sleep 20
echo "[post] eval start $(date)"
python eval.py --ckpt $OUT/ckpt_final.pt --latency --out $OUT/eval_results.json --save_rows $OUT/eval_rows.json
echo "[post] baseline start $(date)"
python baseline.py --model Qwen/Qwen3.5-4B --n 500 --out $OUT/baseline_results.json
python scripts/export_ckpt.py $OUT/ckpt_final.pt $OUT/ckpt_final_bf16.pt
echo "[post] ALL DONE $(date)"
