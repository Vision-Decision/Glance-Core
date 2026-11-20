#!/usr/bin/env bash
# Second generation VM: COCO train2014 + vLLM venv, then catfree_gen on a scene split; pushes the output to HF when done.
# Usage (VM): nohup bash vm_gen2.sh /content/v12/scenes_b.jsonl gen_b.jsonl > /content/v12_gen2.log 2>&1 &
set -uo pipefail
SC=$1; OUTN=$2
mkdir -p /content/data/coco && cd /content/data/coco
[ -f train2014/.done ] || { wget -q http://images.cocodataset.org/zips/train2014.zip && unzip -q -o train2014.zip && rm -f train2014.zip && touch train2014/.done; }
echo "[gen2] coco $(ls train2014 | wc -l) $(date -u)"
bash /content/dv/scripts/v12/vm_vllm.sh > /content/v12_vllm.log 2>&1 || { echo "[gen2] vllm setup failed"; exit 1; }
cd /content/dv/scripts/v12
HF_HUB_OFFLINE=1 /content/venv_vllm/bin/python catfree_gen.py --scenes $SC --img_root /content/data/coco/train2014 --out /content/v12/$OUTN
echo "[gen2] gen exit rc=$? $(date -u)"
python -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/content/v12/$OUTN', path_in_repo='v1.2/data/$OUTN', repo_id='Vision-Decision/glance-train-ckpts', repo_type='model')" \
  && echo "[gen2] pushed $(date -u)"
