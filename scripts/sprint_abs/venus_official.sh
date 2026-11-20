#!/usr/bin/env bash
# Run the OFFICIAL VenusBench-GD evaluator (UI-Venus eval.py, unmodified) over all six tasks for one wrapper config.
# Usage (VM): TAG=v1 CFG=stack_gated GLANCE_STACK=/content/sabs/data/stack_gui_v1.json bash venus_official.sh
#   CFG names the output dir; the wrapper env (GLANCE_THR / GLANCE_STACK / GLANCE_GATED) selects the decision rule.
set -u
TAG=${TAG:-v1}; CFG=${CFG:-raw_gated}; S=/content/sabs; VR=/content/gui/venus_raw
OUT=$S/venus_$TAG/$CFG; mkdir -p $OUT
export GLANCE_CKPT=${GLANCE_CKPT:-/content/ckpt/$TAG/ckpt.pt} GLANCE_ANNS=$VR/instruction GLANCE_IMAGES=$VR/images \
  GLANCE_CACHE=${GLANCE_CACHE:-$S/venus_$TAG/cache.json} DV_ROOT=/content/dv PYTHONPATH=/content/dv
cp /content/dv/scripts/sprint_abs/venus_glance.py /content/uivenus/models/glance.py
cd /content/uivenus
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
print("$CFG", {k: round(100 * v[0] / v[1], 2) for k, v in r.items()}, "overall", round(100 * c / n, 2), n)
PY
