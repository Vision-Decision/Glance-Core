#!/usr/bin/env bash
# Re-run the training-pool prep after the GUI eval sets exist (eval-image hashes must include them for dedup).
cd /content/dv
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
pkill -f "prep_all.sh" ; pkill -f "dv.data.prep_v1" ; pkill -f "sprint_robust/vm_setup.sh"; sleep 3
[ -f /content/evals.tgz ] && mkdir -p /content/dv/evals && tar xzf /content/evals.tgz -C /content/dv
python evals/gui_data.py --root /content/gui > /content/gui_data.log 2>&1; echo "[fix] gui_data rc=$? $(ls /content/gui)"
O=/content/data/prep_v1; mkdir -p $O; cp -n /content/up/pool_natural.jsonl /content/up/calib_natural.jsonl $O/
rm -f $O/eval_hashes.json
bash scripts/v1/prep_all.sh > /content/prep_all.log 2>&1; echo "[fix] prep_all rc=$?"
python -m dv.data.prep_v1 postdedup --out $O > $O/log_postdedup.txt 2>&1; echo "[fix] postdedup rc=$?"
python -m dv.data.prep_v1 mix --out $O --total 1000000 > $O/log_mix.txt 2>&1; echo "[fix] mix rc=$?"
wc -l $O/*.jsonl; tail -3 $O/log_mix.txt
echo "[fix] DONE $(date -u)"
