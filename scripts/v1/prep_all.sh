#!/usr/bin/env bash
# v1 data prep on the VM (after fetch.sh). Pools only; run the `mix` step after checking the stats.
cd /content/dv
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
O=/content/data/prep_v1; mkdir -p $O
python -m dv.data.prep_v1 text --out $O > $O/log_text.txt 2>&1 &
python -m dv.data.prep_v1 evalhash --out $O > $O/log_evalhash.txt 2>&1
python -m dv.data.prep_v1 gui --out $O > $O/log_gui.txt 2>&1 &
python -m dv.data.prep_v1 pixmo --out $O > $O/log_pixmo.txt 2>&1
wait
echo "[prep_all] DONE $(date)"
