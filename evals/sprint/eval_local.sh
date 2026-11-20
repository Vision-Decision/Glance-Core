#!/usr/bin/env bash
# eval_local.sh CKPT TAG RUNS : like eval_ckpt.sh but for a checkpoint already on this VM (e.g. sg-train after training).
cd /content/dv
CK=$1; TAG=$2; RUNS=$3
until [ -f $CK ]; do sleep 30; done
bash evals/sprint/queue_inf.sh $CK $TAG "$RUNS"
bash evals/sprint/zoom_queue.sh $CK /content/sprint/${TAG}_sspro_16384u1 sspro "0.25:4096:2:box 0.25:4096:2:box:3"
python evals/sprint/analyze.py --runs $(ls /content/sprint/${TAG}_*u1.info.json | sed 's/.info.json//' | paste -sd, -) \
  --zoom /content/sprint/${TAG}_sspro_16384u1=/content/sprint/${TAG}_sspro_16384u1_zoom0.25_4096u2box \
  --mzoom /content/sprint/${TAG}_sspro_16384u1=/content/sprint/${TAG}_sspro_16384u1_zoom0.25_4096u2boxc3 --out /content/sprint/an_${TAG}.json
python evals/sprint/collect.py --dir /content/sprint --out /content/sprint/sprint_gui_results_${TAG}.json
echo "[q] ALL DONE eval_local $TAG $(date -u +%T)"
