#!/usr/bin/env bash
# v1.1 prep chain on the VM: text/evalhash/pixmo once fetch.sh is done; gui + postdedup once fetch_v11.sh is done.
cd /content/dv
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
O=/content/data/prep_v1
until grep -q "\[fetch\].*DONE" /content/fetch.log 2>/dev/null; do sleep 30; done
python -m dv.data.prep_v1 text --out $O > $O/log_text.txt 2>&1 &
python -m dv.data.prep_v1 evalhash --out $O > $O/log_evalhash.txt 2>&1
python -m dv.data.prep_v1 pixmo --out $O --pixmo_point_images 20000 --pixmo_count_images 20000 > $O/log_pixmo.txt 2>&1 &
until grep -q "\[f11\].*DONE" /content/fetch11.log 2>/dev/null; do sleep 30; done
python -m dv.data.prep_v1 gui --out $O > $O/log_gui.txt 2>&1
wait
python -m dv.data.prep_v1 postdedup --out $O > $O/log_postdedup.txt 2>&1
echo "[prep_v11] DONE $(date)"
