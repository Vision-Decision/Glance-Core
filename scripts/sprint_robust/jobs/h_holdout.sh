#!/usr/bin/env bash
# held-out affordance categories: functional/generic rewrites never target cup,wine glass,donut,cake,couch,bed,laptop,cell phone,umbrella,horse,clock,toaster
while pgrep -f "[b]ase_gen.py|[e]val.py|[t]rain.py" > /dev/null; do sleep 30; done
export NAME=h_holdout STEPS=${STEPS:-800} VARIANTS=zh,para,func,ufunc,ugeneric,absfunc,absgeneric TRAIN_SET="w_cons=1.0" MIX_ARGS="--aug /content/up/aug_holdout.jsonl" SETS=rg,robust_func,robust_vague,refcoco_val,grefcoco_val_merged
bash /content/dv/scripts/sprint_robust/run_recipe.sh
