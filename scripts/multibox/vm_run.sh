#!/usr/bin/env bash
# VM side of the v1.2 multi-box pilot. Code + data arrive as /content/dv.tgz (scripts/multibox/push.sh).
# Usage (VM): bash /content/dv/scripts/multibox/vm_run.sh MODES > /content/mb.log 2>&1     (MODES comma-separated, run in order)
#   joint | heads : smoke test, train configs/v1.2_pilot_MODE.json, eval, push to HF
#   base          : eval the v1.1 checkpoint on the same sets, push to HF
# Results: /content/out/mb_MODE/{eval.json,rows/}; HF Vision-Decision/glance-train-ckpts-legacy v1.2_pilot/MODE/
set -uo pipefail
MODES=${1:?modes}; REPO=Vision-Decision/glance-train-ckpts-legacy
cd /content/dv
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
[ -f /content/.hf_token ] && export HF_TOKEN=$(cat /content/.hf_token)
echo "[mb] $(date -u) setup"
bash scripts/colab_setup.sh > /content/setup.log 2>&1 || { echo "setup failed"; tail -20 /content/setup.log; exit 1; }
mkdir -p /content/data/prep_v1 /content/data/prep_mb /content/ckpt/v1.1
cp data/prep_v1/calib_natural.jsonl /content/data/prep_v1/
cp data/prep_mb/*.jsonl /content/data/prep_mb/
[ -f /content/ckpt/v1.1/ckpt_final.pt ] || hf download $REPO v1.1/ckpt_final.pt --local-dir /content/ckpt >/dev/null
ls -la /content/ckpt/v1.1/
# calib_mb (v1 records) -> eval format, so eval.py can score it as set "calibmb" (threshold sanity only; not an eval set)
python - <<'PY'
import json
with open("/content/dv/data/prep/eval_calibmb.jsonl", "w") as f:
    for l in open("/content/data/prep_mb/calib_mb.jsonl"):
        r = json.loads(l)
        f.write(json.dumps(dict(image_id=r["image_id"], file=r["img"].split("/")[-1], W=r["wh"][0], H=r["wh"][1],
                                questions=r["questions"])) + "\n")
PY
SETS=refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test,grec_val,cocoall_val,calibmb
EV="--calib_dir /content/data/prep_v1 --prep_dir /content/dv/data/prep --sets $SETS"
for MODE in ${MODES//,/ }; do
OUT=/content/out/mb_$MODE; mkdir -p $OUT
if [ "$MODE" = base ]; then
  CK=/content/ckpt/v1.1/ckpt_final.pt
else
  CFG=configs/v1.2_pilot_$MODE.json
  IC=$(python -c "import json;print(json.load(open('$CFG')).get('init_ckpt',''))")
  if [ ! -f "$IC" ] && [[ "$IC" == */mb_heads/* ]]; then  # joint stage on another VM: heads-stage ckpt from HF
    mkdir -p /content/out/mb_heads && hf download $REPO v1.2_pilot/heads/ckpt_final.pt --local-dir /content/hfck >/dev/null \
      && cp /content/hfck/v1.2_pilot/heads/ckpt_final.pt "$IC"
  fi
  echo "[mb] $(date -u) smoke"
  python train.py --cfg $CFG --set limit_records=40 max_steps=3 warmup_steps=1 out_dir=/content/out/smoke_$MODE save_every_min=0 \
    > /content/out/smoke_$MODE.log 2>&1 || { echo "smoke train failed"; tail -40 /content/out/smoke_$MODE.log; continue; }
  tail -3 /content/out/smoke_$MODE.log
  python eval.py --ckpt /content/out/smoke_$MODE/ckpt_final.pt --calib_dir "" --prep_dir /content/dv/data/prep --sets calibmb \
    --out /content/out/smoke_$MODE/eval.json > /content/out/smoke_$MODE/eval.log 2>&1 || { echo "smoke eval failed"; tail -40 /content/out/smoke_$MODE/eval.log; continue; }
  tail -2 /content/out/smoke_$MODE/eval.log | cut -c1-600
  echo "[mb] $(date -u) train"
  nohup bash scripts/v1/push_ckpt_hf.sh $OUT $REPO v1.2_pilot/$MODE 900 > /content/push_$MODE.log 2>&1 &
  python train.py --cfg $CFG > $OUT/train.log 2>&1 || { echo "train failed"; tail -40 $OUT/train.log; continue; }
  tail -2 $OUT/train.log
  CK=$OUT/ckpt_final.pt
fi
echo "[mb] $(date -u) eval"
python eval.py --ckpt $CK $EV --out $OUT/eval.json --rows_dir $OUT/rows > $OUT/eval.log 2>&1 || { echo "eval failed"; tail -40 $OUT/eval.log; }
echo "[mb] $(date -u) push"
cp $OUT/train_log.jsonl $OUT/rows/ 2>/dev/null; cp $OUT/eval.json $OUT/rows/
timeout 3600 hf upload $REPO $OUT/rows v1.2_pilot/$MODE/eval --repo-type model --commit-message "v1.2 pilot $MODE eval" >/dev/null 2>&1 && echo pushed
echo "[mb] $(date -u) $MODE DONE"
done
sleep 120  # let push_ckpt_hf.sh finish the final checkpoints
echo "[mb] $(date -u) ALL DONE"
