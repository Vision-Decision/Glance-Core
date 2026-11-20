#!/usr/bin/env bash
# Official VenusBench-GD evaluator (UI-Venus eval.py, branch VenusBench-GD, unmodified) on a precomputed wrapper cache
# (no GPU needed: the wrapper never loads a model when GLANCE_CACHE exists). Same loop as scripts/sprint_abs/venus_official.sh.
# Usage: OUT=dir CACHE=cache.json [GLANCE_GATED=1] bash gen_v1/venus_official.sh   (UI-Venus at $UV, data at $VR)
set -u
UV=${UV:-/content/uivenus}; VR=${VR:-/content/gui/venus_raw}; mkdir -p $OUT
export GLANCE_CACHE=$CACHE GLANCE_ANNS=$VR/instruction GLANCE_IMAGES=$VR/images GLANCE_GATED=${GLANCE_GATED:-1} DV_ROOT=${DV_ROOT:-/content/dv}
cd $UV
for task in element_grounding spatial_grounding visual_grounding reason_grounding functional_grounding; do
  [ -f $OUT/$task.json ] || python eval.py --model_type glance --images_dir $VR/images --anns_dir $VR/instruction --task $task \
    --language en --gt_type positive --inst_style instruction --log_path $OUT/$task.json > $OUT/$task.log 2>&1
done
[ -f $OUT/refusal_grounding.json ] || python eval.py --model_type glance --images_dir $VR/images --anns_dir $VR/instruction \
  --task refusal_grounding --language en --gt_type negative --inst_style instruction --log_path $OUT/refusal_grounding.json > $OUT/refusal.log 2>&1
python - <<PY
import json, glob, os
r = {}
for f in sorted(glob.glob("$OUT/*.json")):
    o = json.load(open(f))["metrics"]["overall"]; r[os.path.basename(f)[:-5]] = (o["num_correct_action"], o["num_total"], o["wrong_format_num"])
c = sum(v[0] for v in r.values()); n = sum(v[1] for v in r.values())
print("$OUT", {k: round(100 * v[0] / v[1], 2) for k, v in r.items()}, "overall", round(100 * c / n, 2), n)
PY
