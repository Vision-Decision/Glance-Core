#!/usr/bin/env bash
# sg-inf1 re-prioritisation: let the running v1 sweep finish SS-Pro@4096, drop its remaining (venus / ss2) runs so the
# zoom queue starts, fetch a fresh v1.1 ckpt_last meanwhile, then run v1.1-last on SS-Pro / SS-v2 native after zoom.
cd /content/dv
mkdir -p /content/ckpt/v1.1_last && timeout 1800 hf download Vision-Decision/glance-train-ckpts-legacy v1.1/ckpt_last.pt --local-dir /content/ckpt/v1.1_last >/dev/null && echo "[q] got fresh v1.1 ckpt_last"
until [ -f /content/sprint/v1_sspro_4096u1.info.json ]; do sleep 20; done
pkill -f '[q]ueue_inf.sh /content/ckpt/v1/'; sleep 2; pkill -f '[i]nfer.py --ckpt /content/ckpt/v1/ckpt_final.pt --bench /content/gui/ss2'
echo "[q] ALL DONE v1 (truncated by reorder) $(date -u +%T)" >> /content/q_v1.log
until grep -q "ALL DONE zoom" /content/q_zoom_v1.log 2>/dev/null; do sleep 20; done
bash evals/sprint/queue_inf.sh /content/ckpt/v1.1_last/v1.1/ckpt_last.pt v11last "sspro:16384:1 ss2:16384:1 ss2:2048:2"
