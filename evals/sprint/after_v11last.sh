#!/usr/bin/env bash
# sg-inf1 after the v1.1-last sweep: zoom on v1.1-last (single + 3 candidates), then v1 cascaded / multi-candidate zoom.
cd /content/dv
until grep -q "ALL DONE v11last" /content/q_v11last.log 2>/dev/null; do sleep 30; done
Z=evals/sprint/zoom_queue.sh
bash $Z /content/ckpt/v1.1_last/v1.1/ckpt_last.pt /content/sprint/v11last_sspro_16384u1 sspro "0.25:4096:2:box 0.25:4096:2:box:3"
bash $Z /content/ckpt/v1/ckpt_final.pt /content/sprint/v1_sspro_16384u1 sspro "0.25:4096:2:box:3"
bash $Z /content/ckpt/v1/ckpt_final.pt /content/sprint/v1_sspro_16384u1_zoom0.4_4096u2box sspro "0.16:4096:2:box"
echo "[q] ALL DONE after_v11last $(date -u +%T)"
