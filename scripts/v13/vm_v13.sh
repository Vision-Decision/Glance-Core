#!/usr/bin/env bash
# v1.3 on the v1.2 VM: wait for v12b + zoom re-measure, pull the v1.3 generated data from HF, build the mixture, push it,
# train (configs/v1.3_4b.json; set by the head-off outcome), then evaluate (PREREG A-C, D via S2; multi-box sets if on).
# Code: /content/dv13.tgz (branch v12b-multibox, which contains everything) is unpacked over /content/dv once the zoom
# re-measure (which uses the old /content/dv) is done.  Usage (VM): nohup bash /content/vm_v13.sh > /content/v13.log 2>&1 &
set -uo pipefail
D=/content/dv; REPO=Vision-Decision/glance-train-ckpts; DATA=Vision-Decision/glance-train-data
export PYTHONPATH=$D HF_HUB_DISABLE_PROGRESS_BARS=1
until grep -q "\[zoom\] ALL DONE" /content/v12_zoom.log 2>/dev/null; do sleep 60; done
tar xzf /content/dv13.tgz -C $D && cd $D
mkdir -p /content/v13 && cp -n /content/dv_mb/data/prep/eval_*.jsonl $D/data/prep/ 2>/dev/null
until python -c "
from huggingface_hub import HfApi; import sys
fs = HfApi().list_repo_files('$DATA', repo_type='dataset')
sys.exit(0 if all(f'v1.3/raw/{x}' in fs for x in ['zh_filtered.jsonl', 'gen_v13.jsonl', 'scenes_v13_10k.jsonl']) else 1)"; do sleep 120; done
python -c "
from huggingface_hub import hf_hub_download
for x in ['zh_trans.jsonl', 'zh_filtered.jsonl', 'gen_v13.jsonl', 'scenes_v13_10k.jsonl']:
    print(hf_hub_download('$DATA', f'v1.3/raw/{x}', repo_type='dataset', local_dir='/content/v13dl'))"
python -c "
from huggingface_hub import hf_hub_download
try:
    print(hf_hub_download('$DATA', 'v1.3/raw/zh_opus.jsonl', repo_type='dataset', local_dir='/content/v13dl', force_download=True))
except Exception as e:
    print('no opus translations yet', type(e).__name__)"
cp /content/v13dl/v1.3/raw/*.jsonl /content/v13/
echo "[v13] build $(date -u)"
python scripts/v12/catfree_build.py --scenes /content/v13/scenes_v13_10k.jsonl --gen /content/v13/gen_v13.jsonl \
  --out_dir /content/v13/out --n_pos 0 --n_neg 200000 > /content/v13/neg_build.log 2>&1 || { echo "neg build failed"; exit 1; }
python scripts/v13/build_mix_v13.py --prep /content/data/prep_v1 --gd /content/gd --catfree12 /content/v12/out/catfree.jsonl \
  --neg13 /content/v13/out/catfree.jsonl --count /content/v13/count_pos.jsonl --zh /content/v13/zh_filtered.jsonl \
  --pool_natural /content/data/prep_v1/pool_natural.jsonl --eval_ids /content/v12/eval_image_ids.json --out /content/data/prep_v13 \
  > /content/v13/build_mix.log 2>&1 || { echo "build_mix failed"; tail -20 /content/v13/build_mix.log; exit 1; }
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for src, dst in [("/content/data/prep_v13/train.jsonl", "v1.3/train.jsonl"), ("/content/data/prep_v13/stats_v13.json", "v1.3/stats_v13.json"),
                 ("/content/v13/out/catfree.jsonl", "v1.3/neg13.jsonl"), ("/content/v13/out/stats.json", "v1.3/neg13_stats.json")]:
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
if python -c "import json,sys; sys.exit(0 if json.load(open('$CFG')).get('multibox') else 1)"; then
  O=$OUT/post_mb; mkdir -p $O
  HF_HUB_OFFLINE=1 python eval.py --ckpt $OUT/ckpt_final.pt --calib_dir /content/data/prep_v1 --prep_dir $D/data/prep --num_workers 4 \
    --out $O/eval.json --rows_dir $O/rows --sets grec_val,cocoall_val,calibmb,diag > $O/eval.log 2>&1
  for M in thr count; do python scripts/v12b/multi_f1.py --rows $O/rows/grec_val.jsonl.gz --mode $M --calib $O/rows/calibmb.jsonl.gz > $O/multi_f1_$M.json; done
  timeout 1800 hf upload $REPO $O v1.3/post_mb --repo-type model > /dev/null 2>&1
fi
echo "[v13] ALL DONE $(date -u)"
