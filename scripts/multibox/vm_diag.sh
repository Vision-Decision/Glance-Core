#!/usr/bin/env bash
# VM side: score v1.1 and the pilot checkpoints on the glance-ground-v2 diagnostic split (300 refcoco_val images,
# Glance-Data wording incl. Chinese; eval only). Waits for vm_run.sh to finish. Per-question qtype / lang go into
# "meta" of the row dumps for a per-type breakdown.
# Usage (VM): bash /content/dv/scripts/multibox/vm_diag.sh [NAME ...] > /content/diag.log 2>&1   (default: all runs)
set -uo pipefail
cd /content/dv
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
[ -f /content/.hf_token ] && export HF_TOKEN=$(cat /content/.hf_token)
REPO=Vision-Decision/glance-train-ckpts-legacy
until grep -q "ALL DONE" /content/mb.log; do sleep 60; done
[ -f /content/gv2/diagnostic.jsonl ] || hf download Vision-Decision/glance-ground-v2 diagnostic.jsonl --repo-type dataset --local-dir /content/gv2 >/dev/null
python - <<'PY'
import json
with open("/content/dv/data/prep/eval_diag.jsonl", "w") as f:
    for l in open("/content/gv2/diagnostic.jsonl"):
        r = json.loads(l)
        qs = [dict(q, meta=dict(qtype=q.get("qtype"), lang=q.get("lang"))) for q in r["questions"]]
        f.write(json.dumps(dict(image_id=r["image_id"], file=r["img"].split("/")[-1], W=r["wh"][0], H=r["wh"][1],
                                questions=qs), ensure_ascii=False) + "\n")
PY
ALL="base:/content/ckpt/v1.1/ckpt_final.pt heads:/content/out/mb_heads/ckpt_final.pt joint:/content/out/mb_joint/ckpt_final.pt jointnb:/content/out/mb_jointnb/ckpt_final.pt jointnb2:/content/out/mb_jointnb2/ckpt_final.pt"
for M in $ALL; do
  N=${M%%:*}; CK=${M#*:}; O=/content/out/diag_$N; mkdir -p $O
  [ $# -eq 0 ] || [[ " $* " == *" $N "* ]] || continue
  [ -f $CK ] || { echo "missing $CK"; continue; }
  echo "[diag] $(date -u) $N"
  python eval.py --ckpt $CK --calib_dir "" --prep_dir /content/dv/data/prep --sets diag --out $O/eval.json --rows_dir $O/rows \
    > $O.log 2>&1 || { echo "eval failed $N"; tail -20 $O.log; continue; }
  cp $O/eval.json $O/rows/
  timeout 1800 hf upload $REPO $O/rows v1.2_pilot/$N/diag --repo-type model --commit-message "v1.2 pilot $N diagnostic" >/dev/null 2>&1 && echo pushed $N
done
echo "[diag] $(date -u) ALL DONE"
