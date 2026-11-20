#!/usr/bin/env bash
# Re-measure ScreenSpot-Pro (single pass + crop-zoom) for v1.1 and v1.2 with the order-preserving predict_v1 fix.
# Waits for the v12b runs (one GPU job at a time). Usage (VM): nohup bash vm_zoom.sh > /content/v12_zoom.log 2>&1 &
set -uo pipefail
until grep -q "\[v12b\] ALL DONE" /content/v12b.log 2>/dev/null; do sleep 60; done
cd /content/dv; export PYTHONPATH=/content/dv HF_HUB_OFFLINE=1
O=/content/out/zoom_fix; mkdir -p $O
for M in v1.1:/content/ckpt/v1.1/ckpt_final.pt v1.2:/content/out/v1_2/ckpt_final.pt; do
  N=${M%%:*}; C=${M#*:}
  python eval.py --ckpt $C --calib_dir /content/data/prep_v1 --out $O/sspro_$N.json --rows_dir $O/rows_$N --num_workers 2 \
    --gui_max_vis 8192 --gui_upscale 1.0 --zoom sspro --zoom_frac 0.4 --zoom_max_vis 4096 --sets sspro > $O/$N.log 2>&1
  echo "[zoom] $N $(python -c "import json;s=json.load(open('$O/sspro_$N.json'))['sets']['sspro'];print(s['raw']['click_acc'],s['zoom']['click_acc'])") $(date -u)"
done
unset HF_HUB_OFFLINE
timeout 1800 hf upload Vision-Decision/glance-train-ckpts $O v1.2/zoom_fix --repo-type model > /dev/null 2>&1 && echo "[zoom] pushed"
echo "[zoom] ALL DONE $(date -u)"
