#!/usr/bin/env bash
# Full sprint GUI eval of one checkpoint from the private HF repo:
#   eval_ckpt.sh HF_PATH TAG [WAIT_LOG] [RUNS]
# native-resolution SS-Pro / SS-v2 / VenusBench-GD (+ SS-v2 at the v1 protocol 2048u2), then SS-Pro crop-and-zoom.
cd /content/dv
HP=$1; TAG=$2; WAIT=${3:-}; RUNS=${4:-"sspro:16384:1 ss2:16384:1 ss2:2048:2 venus:16384:1"}
D=/content/ckpt/$TAG; mkdir -p $D
timeout 1800 hf download Vision-Decision/glance-train-ckpts-legacy $HP --local-dir $D >/dev/null && echo "[q] got $HP as $TAG $(date -u +%T)"
WAITPAT="ALL DONE"; [[ $WAIT == *@* ]] && { WAITPAT=${WAIT#*@}; WAIT=${WAIT%%@*}; }   # WAIT_LOG@pattern
[ -n "$WAIT" ] && until grep -q "$WAITPAT" $WAIT 2>/dev/null; do sleep 30; done
bash evals/sprint/queue_inf.sh $D/$HP $TAG "$RUNS"
bash evals/sprint/zoom_queue.sh $D/$HP /content/sprint/${TAG}_sspro_16384u1 sspro "0.25:4096:2:box 0.25:4096:2:box:3"
echo "[q] ALL DONE eval_ckpt $TAG $(date -u +%T)"
