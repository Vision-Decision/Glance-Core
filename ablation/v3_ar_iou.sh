#!/usr/bin/env bash
# ablation_v3 AR IoU head (VM side, cwd /content/dv). Idempotent; log in /content/out/v3_ar_iou.log.
#  1 env + COCO (scripts/colab_setup.sh)   2 pull bf16 ckpts (ar_digits_neg25, ours_neg25) from the private HF repo
#  3 frozen AR-digits: constrained greedy rows + box features on test (gRefCOCO val merged), C (arcal_gtest),
#    H (arhead_rcg), T (arhead_train)        (ablation/ar_iou_feats.py)
#  4 gate (ours_neg25) rows on C             (ablation/eval_ab.py)
#  5 IoU heads on the frozen AR features     (ablation/ar_iou_head.py)
#  6 pack small outputs (/content/out/v3_small.tgz); push rows to HF Vision-Decision/glance-train-ckpts baselines/ar_iou/ (old checkpoints are read from Vision-Decision/glance-train-ckpts-legacy)
cd /content/dv
export PYTHONUNBUFFERED=1
O=/content/out/v3; mkdir -p $O
LOG=/content/out/v3_ar_iou.log
log() { echo "[v3] $* $(date +%H:%M:%S)" >> $LOG; }
log start
[ -f /content/setup.done ] || { bash scripts/colab_setup.sh > /content/setup.log 2>&1 && touch /content/setup.done || { log SETUP FAIL; exit 1; }; }
log setup ok
for R in ar_digits_neg25 ours_neg25; do
  [ -f $O/$R.pt ] || hf download Vision-Decision/glance-train-ckpts-legacy ablation/${R}_bf16.pt --local-dir $O/hf > /dev/null 2>&1 \
    && mv $O/hf/ablation/${R}_bf16.pt $O/$R.pt
  [ -f $O/$R.pt ] || { log CKPT FAIL $R; exit 1; }
done
log ckpts ok
python -m ablation.ar_iou_feats --ckpt $O/ar_digits_neg25.pt --sets grefcoco_val_merged,arcal_gtest,arhead_rcg,arhead_train \
  --out_dir $O/feats >> $O/feats_stdout.log 2>&1 || { log FEATS FAIL; exit 1; }
log feats ok
[ -f $O/ours_rows_arcal.json ] || python -m ablation.eval_ab --ckpt $O/ours_neg25.pt --sets arcal_gtest --out $O/ours_eval_arcal.json \
  --rows $O/ours_rows_arcal.json > $O/ours_stdout.log 2>&1 || { log OURS FAIL; exit 1; }
log ours ok
python -m ablation.ar_iou_head --feat_dir $O/feats --out $O/head_scores.npz > $O/head_stdout.log 2>&1 || { log HEAD FAIL; exit 1; }
log head ok
cd $O && tar czf /content/out/v3_small.tgz head_scores.npz head_scores_info.json head_stdout.log feats_stdout.log ours_stdout.log \
  ours_eval_arcal.json ours_rows_arcal.json feats/rows_*.json && cd /content/dv
log packed
timeout 1800 hf upload Vision-Decision/glance-train-ckpts /content/out/v3_small.tgz baselines/ar_iou/ar_iou_rows.tgz --repo-type model \
  --commit-message "ablation_v3 AR IoU head rows" > /dev/null 2>&1 && log pushed || log PUSH FAIL
log ALL DONE
# features too (re-fit heads later without a GPU); best effort
timeout 2400 hf upload Vision-Decision/glance-train-ckpts $O/feats baselines/ar_iou/ar_iou_feats --repo-type model \
  --commit-message "ablation_v3 AR features" > /dev/null 2>&1 && log feats pushed || log FEATS PUSH FAIL
log FINISHED
