#!/usr/bin/env bash
# Full ScreenSpot-Pro zoom / reduced-prefix runs (exps 2 and 8). Usage (VM): bash arch/zoom/queue.sh CKPT "spec ..."
# spec = prefix_vis:crop_vis:layout:keep_ratio:extra   (extra = rp for --reprefill, '-' otherwise)
CK=$1; SPECS=$2
cd /content/dv; mkdir -p /content/arch/zoom
for s in $SPECS; do
  IFS=: read PV CV LY KR EX AL <<< "$s"; AL=${AL:-1.0}
  O=/content/arch/zoom/sspro_p${PV}_c${CV}_${LY}_k${KR}; [ "$AL" != "1.0" ] && O=${O}_a${AL}
  [ -f $O.json ] && continue
  X=""; [ "$EX" = rp ] && X="--reprefill"
  echo "[q] $(date -u +%T) $O"
  python -m arch.zoom.run_zoom --ckpt $CK --prefix_vis $PV --crop_vis $CV --layout $LY --keep_ratio $KR --alpha $AL $X --out $O > $O.log 2>&1
  grep -a "DONE\|Error\|error" $O.log | tail -2
done
echo "[q] ZOOM_QUEUE_DONE $(date -u +%T)"
