set -e
cd /content/amb
pip install -q --target /content/amb/pylib "glance @ git+https://github.com/Vision-Decision/Glance" > pip.log 2>&1
export PYTHONPATH=/content/amb/pylib
mkdir -p img
python -c "
import json
for l in open('eval_grefcoco_val_merged.jsonl'): print('http://images.cocodataset.org/train2014/'+json.loads(l)['file'])
" | sort -u > urls.txt
(cd img && xargs -n1 -P32 wget -q -nc < ../urls.txt) || true
ls img | wc -l > nimg.txt
nohup python run_grefcoco.py eval_grefcoco_val_merged.jsonl img grefcoco_full.json > run.log 2>&1 &
echo launched
