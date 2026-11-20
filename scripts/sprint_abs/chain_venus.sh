#!/usr/bin/env bash
# After the v1 pipeline (which builds the VenusBench forward cache), score wrapper configs with the official evaluator.
#   raw_gated:   p = sigmoid(presence logit), refuse iff p < 0.5, a refused item returns no point (faithful mapping)
#   platt_gated: p = Platt(presence logit) fitted on guisyn-fit apps only, refuse iff p < 0.5, gated
#   stack_gated: 5-feature recalibrator fitted on guisyn-fit (reported because it is the best on held-out guisyn apps)
#   raw_ungated: the point is always returned (NOT faithful; only to reproduce the earlier own-script number)
cd /content/dv
TAG=${TAG:-v1}
until grep -q "\[sabs\].*DONE" /content/sabs/run_$TAG.log 2>/dev/null; do sleep 30; done
S=/content/sabs
CFG=raw_gated bash scripts/sprint_abs/venus_official.sh
CFG=platt_gated GLANCE_STACK=$S/data/platt_gui_$TAG.json bash scripts/sprint_abs/venus_official.sh
CFG=stack_gated GLANCE_STACK=$S/data/stack_gui_$TAG.json bash scripts/sprint_abs/venus_official.sh
CFG=raw_ungated GLANCE_GATED=0 bash scripts/sprint_abs/venus_official.sh
timeout 1800 hf upload Vision-Decision/glance-train-ckpts $S/venus_$TAG sprint_abs/venus_$TAG --repo-type model --exclude "cache.json" \
  --commit-message "sprint_abs venus official $TAG" >/dev/null 2>&1 && echo pushed
echo "[chain_venus] DONE $(date -u)"
