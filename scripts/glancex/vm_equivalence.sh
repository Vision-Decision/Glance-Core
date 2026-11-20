#!/usr/bin/env bash
# dv.glancex vs upstream glancex on the released checkpoint, all three directions, same subsets and batching.
# Usage (VM): bash /content/dv/scripts/glancex/vm_equivalence.sh > /content/gx/equiv.log 2>&1
set -uo pipefail
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
cd /content/dv
E=/content/gx/equiv
for MODE in image_first text_first joint; do
  for IMPL in glancex dv; do
    python scripts/glancex/dump_rows.py --src Vision-Decision/Glance-X-4B --mode $MODE --impl $IMPL --limit_images 48 \
      --sets refcoco_val,grefcoco_testA,diag --diag /content/gxdata/diagnostic.jsonl --out $E/$IMPL/$MODE
  done
  python scripts/glancex/compare_rows.py $E/glancex/$MODE $E/dv/$MODE $E/compare_$MODE.json
done
python - <<'PY'
import json
r = {m: json.load(open(f"/content/gx/equiv/compare_{m}.json")) for m in ("image_first", "text_first", "joint")}
json.dump(r, open("/content/gx/equiv/equivalence.json", "w"), indent=1)
print("ALL_EQUAL" if all(v["bitwise_equal"] for m in r.values() for v in m.values()) else "NOT_BITWISE", flush=True)
PY
timeout 900 hf upload Vision-Decision/glance-train-ckpts $E/equivalence.json results/glancex/equivalence.json --repo-type model >/dev/null 2>&1 && echo pushed
