#!/usr/bin/env bash
# v1.2 chain as actually run (after the data build): train 3800 steps, post-training eval (PREREG A-C), group D (S2 pipeline).
# vm_v12b.sh (branch v12b-multibox) waits for this script's "[chain2] ALL DONE" line in /content/v12_chain2.log.
# Usage (VM): nohup bash /content/dv/scripts/v12/vm_chain2.sh > /content/v12_chain2.log 2>&1 &
cd /content/dv
bash scripts/v12/vm_train.sh max_steps=3800 > /content/v12_train.log 2>&1
bash scripts/v12/vm_post.sh > /content/v12_post.log 2>&1
bash scripts/v12/vm_post_s2.sh > /content/v12_post_s2.log 2>&1
echo "[chain2] ALL DONE $(date -u)"
