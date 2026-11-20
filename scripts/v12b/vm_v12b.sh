#!/usr/bin/env bash
# v1.2b = multi-box heads on top of v1.2 (pilot recipe), on the v1.2 VM after the v1.2 evaluation.
# Code: /content/dv_mb (branch v12b-multibox). Needs /content/v12b/{train_mb,calib_mb}.jsonl and
# /content/dv_mb/data/prep/eval_cocoall_val.jsonl (scripts/multibox/prep_pilot.py, built from annotations off the VM).
# Usage (VM): nohup bash /content/dv_mb/scripts/v12b/vm_v12b.sh > /content/v12b.log 2>&1 &
set -uo pipefail
REPO=Vision-Decision/glance-train-ckpts; D=/content/dv_mb
cd $D; export PYTHONPATH=$D HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
until grep -q "\[chain2\] ALL DONE" /content/v12_chain2.log 2>/dev/null; do sleep 60; done
echo "[v12b] start $(date -u)"
mkdir -p $D/data/prep && cp -n /content/dv/data/prep/eval_*.jsonl $D/data/prep/ && cp /content/v12b/eval_cocoall_val.jsonl $D/data/prep/
python -c "from huggingface_hub import snapshot_download as s; s('Vision-Decision/glance-ground-v2', repo_type='dataset', local_dir='/content/gd', allow_patterns=['full/*'])"
python scripts/v12b/build_mix_b.py --pilot /content/v12b/train_mb.jsonl --gd_full /content/gd/full --v12 /content/data/prep_v12/train.jsonl \
  --eval_ids /content/v12/eval_image_ids.json --out /content/data/prep_v12b > /content/v12b/build.log 2>&1 || { echo build failed; exit 1; }
tail -5 /content/v12b/build.log
python - <<'PY'
import json  # calib_mb -> eval format (set "calibmb": only for choosing the decode threshold, never reported as eval)
with open("/content/dv_mb/data/prep/eval_calibmb.jsonl", "w") as f:
    for l in open("/content/v12b/calib_mb.jsonl"):
        r = json.loads(l)
        f.write(json.dumps(dict(image_id=r["image_id"], file=r["img"].split("/")[-1], W=r["wh"][0], H=r["wh"][1], questions=r["questions"])) + "\n")
with open("/content/dv_mb/data/prep/eval_diag.jsonl", "w") as f:
    for l in open("/content/gd/diagnostic.jsonl"):
        r = json.loads(l)
        qs = [dict(q, meta=dict(qtype=q.get("qtype"), lang=q.get("lang"))) for q in r["questions"]]
        f.write(json.dumps(dict(image_id=r["image_id"], file=r["img"].split("/")[-1], W=r["wh"][0], H=r["wh"][1], questions=qs), ensure_ascii=False) + "\n")
PY
for ARM in heat obj; do
  P=v1.2b; [ $ARM = obj ] && P=v1.2b_obj
  for ST in heads joint; do
    CFG=configs/${P}_$ST.json; OUT=$(python -c "import json;print(json.load(open('$CFG'))['out_dir'])")
    python train.py --cfg $CFG --set limit_records=40 max_steps=3 warmup_steps=1 out_dir=/content/out/smoke_${ARM}_$ST save_every_min=0 \
      > /content/out/smoke_${ARM}_$ST.log 2>&1 || { echo "smoke $ARM $ST failed"; tail -30 /content/out/smoke_${ARM}_$ST.log; continue 2; }
    echo "[v12b] train $ARM $ST $(date -u)"
    python train.py --cfg $CFG > /content/v12b/train_${ARM}_$ST.log 2>&1 || { echo "train $ARM $ST failed"; tail -30 /content/v12b/train_${ARM}_$ST.log; continue 2; }
    for f in ckpt_final.pt train_log.jsonl config.json; do
      timeout 1800 hf upload $REPO $OUT/$f $P/$ST/$f --repo-type model > /dev/null 2>&1 && echo "[v12b] pushed $P/$ST/$f"
    done
  done
  C=$OUT/ckpt_final.pt; O=$OUT/post; mkdir -p $O
  echo "[v12b] eval $ARM $(date -u)"
  python eval.py --ckpt $C --calib_dir /content/data/prep_v1 --prep_dir $D/data/prep --num_workers 4 --out $O/eval.json --rows_dir $O/rows \
    --sets refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test,grefcoco_val_merged,grec_val,cocoall_val,calibmb,diag \
    > $O/eval.log 2>&1 || tail -30 $O/eval.log
  python scripts/multibox/analyze_rows.py --calib $O/rows/calibmb.jsonl.gz --sets $O/rows/grec_val.jsonl.gz,$O/rows/cocoall_val.jsonl.gz,$O/rows/diag.jsonl.gz \
    > $O/analyze.txt 2>&1
  for M in thr count; do python scripts/v12b/multi_f1.py --rows $O/rows/grec_val.jsonl.gz --mode $M --calib $O/rows/calibmb.jsonl.gz > $O/multi_f1_$M.json 2>&1; done
  python scripts/ground_v2/eval_diag.py --summarize $O/rows/diag.jsonl.gz --out $O/diag_single_view.json > /dev/null 2>&1
  cp $O/eval.json $O/analyze.txt $O/multi_f1_*.json $O/diag_single_view.json $O/rows/ 2>/dev/null
  timeout 1800 hf upload $REPO $O/rows $P/post --repo-type model > /dev/null 2>&1 && echo "[v12b] pushed $P eval"
done
echo "[v12b] ALL DONE $(date -u)"
