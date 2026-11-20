#!/usr/bin/env bash
# After P2: v1.2 alpha sweep (the first attempt failed on a config key) and VenusBench-GD at alpha 1 / .7 / .5 for v1.1
# (does the GUI-path alpha hurt refusal?).  Rows -> /content/arch/alpha, pushed to the org dataset.
cd /content/dv
[ -f /content/gui/venus.jsonl ] || python evals/gui_data.py --root /content/gui --only venus
python -c "from huggingface_hub import hf_hub_download as h; print(h('Vision-Decision/glance-train-ckpts','v1.2/ckpt_final.pt',local_dir='/content/ckpt/v12'))"
python -m arch.alpha.run --ckpt /content/ckpt/v12/v1.2/ckpt_final.pt --alphas 1,0.7,0.5 --what coco --out /content/arch/alpha/v1.2
python -m arch.alpha.run --ckpt /content/ckpt/v12/v1.2/ckpt_final.pt --alphas 1,0.7,0.5 --what sspro --out /content/arch/alpha/v1.2
python -m arch.alpha.run --ckpt /content/ckpt/glance-train-ckpts/v1.1/ckpt_final.pt --alphas 1,0.7,0.5 --what sspro --bench /content/gui/venus.jsonl --out /content/arch/alpha/v1.1_venus
python arch/vm/push.py /content/arch/alpha arch/alpha
echo QB2_DONE
