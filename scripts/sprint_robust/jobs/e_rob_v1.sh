#!/usr/bin/env bash
# glance-robust-ref sets (other agent's eval set) for v1 (Glance heads) and the base generative model
cd /content/dv; export PYTHONPATH=/content/dv TQDM_DISABLE=1
D=/content/out/sr/diag
python eval.py --ckpt /content/ckpt/v1/ckpt_final.pt --calib_dir /content/data/prep_v1 --prep_dir /content/dv/data/prep \
  --sets robust_zh,robust_para,robust_func,robust_vague --rows_dir $D/v1 --out $D/v1_eval_rob.json > $D/v1_eval_rob.log 2>&1
python scripts/sprint_robust/base_gen.py --files /content/dv/data/prep/eval_robust_zh.jsonl,/content/dv/data/prep/eval_robust_para.jsonl,/content/dv/data/prep/eval_robust_func.jsonl,/content/dv/data/prep/eval_robust_vague.jsonl \
  --out_dir $D/base > $D/base_rob.log 2>&1
echo "[e_rob_v1] DONE $(date -u)"
