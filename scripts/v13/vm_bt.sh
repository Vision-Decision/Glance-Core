#!/usr/bin/env bash
# v13 generation VM: stop the negatives generation at N images (enough for the planned counts), let vm_gen13.sh push its
# outputs, then back-translate the Qwen Chinese translations for the round-trip filter and push zh_bt.jsonl.
# Usage (VM): nohup bash vm_bt.sh 7000 > /content/v13_bt.log 2>&1 &
N=${1:-7000}
until [ "$(wc -l < /content/v13/gen_v13.jsonl 2>/dev/null || echo 0)" -ge "$N" ] || ! pgrep -f "catfree_gen.py --scenes /content/v13" > /dev/null; do sleep 30; done
pkill -f "catfree_gen.py --scenes /content/v13"; sleep 5; pkill -9 -f "VLLM::EngineCore"; sleep 10
until grep -q "\[g13\] ALL DONE" /content/v13_gen.log; do sleep 20; done
echo "[bt] gen stopped at $(wc -l < /content/v13/gen_v13.jsonl) images $(date -u)"
python -c "from huggingface_hub import hf_hub_download as d; print(d('Vision-Decision/glance-train-data', 'v1.3/raw/zh_opus.jsonl', repo_type='dataset', local_dir='/content/v13dl'))"
cd /content/dv/scripts/v13
HF_HUB_OFFLINE=1 /content/venv_vllm/bin/python backtrans.py --zh /content/v13/zh_trans.jsonl --skip_keys /content/v13dl/v1.3/raw/zh_opus.jsonl \
  --out /content/v13/zh_bt.jsonl --gpu_mem 0.95 --max_num_seqs 64
python -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/content/v13/zh_bt.jsonl', path_in_repo='v1.3/raw/zh_bt.jsonl', repo_id='Vision-Decision/glance-train-data', repo_type='dataset')" && echo "[bt] pushed"
echo "[bt] ALL DONE $(date -u)"
