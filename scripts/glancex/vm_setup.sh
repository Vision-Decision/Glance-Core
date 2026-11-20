#!/usr/bin/env bash
# Fresh G4 VM -> deps, Qwen3.5-4B(-Base), COCO train2014, upstream Glance-X @ 73ee4ed (pip -e, no deps), the Glance-X
# data mix and diagnostic set (HF Vision-Decision/glance-train-data:glancex/data, built by scripts/glancex/prep_data.py).
# Expects /content/dv (scripts/glancex/sync.sh) and /root/.cache/huggingface/token. Idempotent.
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1
[ -f /content/data/coco/train2014/.done ] || bash /content/dv/scripts/colab_setup.sh > /content/setup.log 2>&1
tail -1 /content/setup.log
if [ ! -d /content/glance-x ]; then
  git clone -q https://github.com/Vision-Decision/Glance-X /content/glance-x
  git -C /content/glance-x checkout -q 73ee4ed
  pip install -q --no-deps -e /content/glance-x
fi
PYTHONPATH=/content/dv python -c "import glancex, dv.glancex; print('glancex', glancex.__file__)"
mkdir -p /content/gxdata
if [ ! -f /content/gxdata/.done ]; then
  hf download Vision-Decision/glance-train-data glancex/data/train_merged.jsonl.gz glancex/data/train_dv_single.jsonl.gz \
     glancex/data/diagnostic.jsonl.gz glancex/data/provenance.json --repo-type dataset --local-dir /content/gxdl >/dev/null
  for f in train_merged train_dv_single diagnostic; do gunzip -c /content/gxdl/glancex/data/$f.jsonl.gz > /content/gxdata/$f.jsonl; done
  cp /content/gxdl/glancex/data/provenance.json /content/gxdata/
  # prep dir for dv's eval.py: our eval sets + the diagnostic set as eval_gxdiag.jsonl
  mkdir -p /content/gxprep && cp /content/dv/data/prep/eval_*.jsonl /content/gxprep/ && cp /content/gxdata/diagnostic.jsonl /content/gxprep/eval_gxdiag.jsonl
  touch /content/gxdata/.done
fi
wc -l /content/gxdata/*.jsonl
echo "[gx-setup] DONE $(date -u)"
