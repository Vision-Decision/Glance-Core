#!/usr/bin/env bash
# v1.3 single-box on a fresh G4: rebuild the v1.1 pools exactly as for v1.2 (vm_setup.sh), pull the v1.3 data from HF, build
# the v1.3 mixture, push it, train configs/v1.3_4b.json from v1.2's ckpt_final, evaluate (PREREG A-C, then D via S2).
# Needs /content/dv (scripts/v12/sync.sh SESSION), HF token, /content/v12/eval_image_ids.json, /content/sabs/data/grefcalib.jsonl.
# Usage (VM): nohup bash /content/dv/scripts/v13/vm_v13_standalone.sh > /content/v13.log 2>&1 &
set -uo pipefail
D=/content/dv; REPO=Vision-Decision/glance-train-ckpts; DATA=Vision-Decision/glance-train-data
cd $D; export PYTHONPATH=$D HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[v13] setup $(date -u)"
NO_VLLM=1 bash scripts/v12/vm_setup.sh > /content/v12_setup.log 2>&1; tail -2 /content/v12_setup.log
mkdir -p /content/v13 /content/v12/out /content/out/v1_2
python - <<'EOF'
import shutil
from huggingface_hub import hf_hub_download as d, snapshot_download as s
s("Vision-Decision/glance-ground-v2", repo_type="dataset", local_dir="/content/gd", allow_patterns=["full/*"])
for f, dst in [("v1.2/catfree.jsonl", "/content/v12/out/catfree.jsonl"), ("v1.3/neg13.jsonl", "/content/v13/neg13.jsonl"),
               ("v1.3/raw/count_pos.jsonl", "/content/v13/count_pos.jsonl"), ("v1.3/raw/zh_filtered.jsonl", "/content/v13/zh_filtered.jsonl")]:
    shutil.copy(d("Vision-Decision/glance-train-data", f, repo_type="dataset", local_dir="/content/gtd"), dst); print("[v13] got", f)
shutil.copy(d("Vision-Decision/glance-train-ckpts", "v1.2/ckpt_final.pt", local_dir="/content/hfck"), "/content/out/v1_2/ckpt_final.pt")
print("[v13] got v1.2 ckpt")
EOF
echo "[v13] build $(date -u)"
python scripts/v13/build_mix_v13.py --prep /content/data/prep_v1 --gd /content/gd --catfree12 /content/v12/out/catfree.jsonl \
  --neg13 /content/v13/neg13.jsonl --count /content/v13/count_pos.jsonl --zh /content/v13/zh_filtered.jsonl \
  --pool_natural /content/data/prep_v1/pool_natural.jsonl --eval_ids /content/v12/eval_image_ids.json --out /content/data/prep_v13 \
  > /content/v13/build_mix.log 2>&1 || { echo "build_mix failed"; tail -20 /content/v13/build_mix.log; exit 1; }
tail -c 3000 /content/v13/build_mix.log
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for src, dst in [("/content/data/prep_v13/train.jsonl", "v1.3/train.jsonl"), ("/content/data/prep_v13/stats_v13.json", "v1.3/stats_v13.json"),
                 ("/content/data/prep_v13/train_base.jsonl", "v1.3/train_base.jsonl")]:
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id="Vision-Decision/glance-train-data", repo_type="dataset")
    print("[v13] pushed", dst, flush=True)
EOF
[ -f /content/v13/HOLD ] && { echo "[v13] HOLD"; while [ -f /content/v13/HOLD ]; do sleep 30; done; }
CFG=configs/v1.3_4b.json; OUT=$(python -c "import json;print(json.load(open('$CFG'))['out_dir'])")
echo "[v13] train $(date -u)"
bash scripts/v12/vm_push.sh $OUT v1.3 > /content/v13_push.log 2>&1 &
HF_HUB_OFFLINE=1 python train.py --cfg $CFG > /content/v13_train.log 2>&1 || { echo "train failed"; tail -30 /content/v13_train.log; exit 1; }
wait
echo "[v13] post $(date -u)"
sed -e "s#/content/out/v1_2#$OUT#g" -e "s#v1.2/post#v1.3/post#g" scripts/v12/vm_post.sh > /content/v13_post.sh
bash /content/v13_post.sh > /content/v13_post.log 2>&1
sed -e "s#/content/out/v1_2#$OUT#g" -e "s#TAG=v1.2#TAG=v1.3#" -e "s#v1.2/post_s2#v1.3/post_s2#" scripts/v12/vm_post_s2.sh > /content/v13_post_s2.sh
bash /content/v13_post_s2.sh > /content/v13_post_s2.log 2>&1
echo "[v13] ALL DONE $(date -u)"
