#!/usr/bin/env bash
# Gen-v1 step 0 (CPU host, e.g. gen-host): rebuild Glance v1's training mixture with the code of the v1 commit (942e648),
# then pack train.jsonl + every GUI / PixMo image it references for the GPU VM (COCO comes from colab_setup.sh there).
# Layout mirrors the v1 VM (/content/dv, /content/data) so the v1 scripts run unchanged.
#   bash prep_gcp.sh  (expects /content -> big disk, /content/dv = git archive 942e648, natural pools copied in, venv active)
set -u
export PYTHONPATH=/content/dv HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
O=/content/data/prep_v1; mkdir -p $O
log() { echo "[gen_v1 prep] $* $(date -u +%T)"; }
log start
# 1. raw data exactly as scripts/v1/fetch.sh @942e648 (HF snapshots, OS-Atlas/AGUVIS zips, eval sets for dedup)
[ -f /content/fetch.done ] || { bash /content/dv/scripts/v1/fetch.sh > /content/fetch.log 2>&1; touch /content/fetch.done; }
log fetch done
# 2. COCO images of the RefCOCO-family eval sets (evalhash needs them; v1 had the full train2014 zip)
python - <<'PY'
import json, glob, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
ids = {json.loads(l)["image_id"] for f in glob.glob("/content/dv/data/prep/eval_*.jsonl") for l in open(f)}
d = "/content/data/coco/train2014"; os.makedirs(d, exist_ok=True)
def get(i):
    p = f"{d}/COCO_train2014_{i:012d}.jpg"
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"http://images.cocodataset.org/train2014/COCO_train2014_{i:012d}.jpg", p)
with ThreadPoolExecutor(32) as ex: list(ex.map(get, sorted(ids)))
print("coco eval images", len(ids))
PY
log coco eval images done
# 3. pools exactly as scripts/v1/prep_all.sh @942e648, then postdedup, then mix (v1: --total 520000, defaults otherwise)
cd /content/dv
[ -s $O/pool_text.jsonl ] || python -m dv.data.prep_v1 text --out $O > $O/log_text.txt 2>&1 &
[ -s $O/eval_hashes.json ] || python -m dv.data.prep_v1 evalhash --out $O > $O/log_evalhash.txt 2>&1
[ -s $O/pool_gui.jsonl ] || python -m dv.data.prep_v1 gui --out $O > $O/log_gui.txt 2>&1 &
[ -s $O/pool_pixmo.jsonl ] || python -m dv.data.prep_v1 pixmo --out $O > $O/log_pixmo.txt 2>&1
wait
log pools done
[ -s $O/stats_postdedup.json ] || python -m dv.data.prep_v1 postdedup --out $O > $O/log_postdedup.txt 2>&1
[ -s $O/train.jsonl ] || python -m dv.data.prep_v1 mix --out $O --total 520000 > $O/log_mix.txt 2>&1
log mix done
# 4. pack: train.jsonl + calib + stats + referenced non-COCO images
python - <<'PY'
import json
imgs = set()
for l in open("/content/data/prep_v1/train.jsonl"):
    r = json.loads(l)
    if r.get("img") and not r["img"].startswith("coco/"):
        imgs.add(r["img"])
open("/content/imglist.txt", "w").write("\n".join(sorted(imgs)) + "\n")
print("non-coco images", len(imgs))
PY
cd /content/data && tar cf /content/genv1_images.tar -T /content/imglist.txt
cd /content/data && tar czf /content/genv1_prep.tgz prep_v1/train.jsonl prep_v1/calib_*.jsonl prep_v1/stats*.json prep_v1/log_*.txt prep_v1/eval_hashes.json
ls -la /content/genv1_images.tar /content/genv1_prep.tgz
log ALL DONE
