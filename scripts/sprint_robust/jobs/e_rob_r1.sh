#!/usr/bin/env bash
# robust-ref sets for r1 (its run started before they were added)
while pgrep -f "[b]ase_gen.py|[e]val.py|[t]rain.py" > /dev/null; do sleep 30; done
export NAME=r1_zhpara_rob EVAL_ONLY=1 INIT=/content/out/sr/r1_zhpara/ckpt_final.pt SETS=robust_zh,robust_para,robust_func,robust_vague
bash /content/dv/scripts/sprint_robust/run_recipe.sh
