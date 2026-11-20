# OVERFIT_OK (v0 box model)

Date: 2026-11-17. No git commits; "known-good" is the working tree under ~/Developer/decision-vison as of this file.

Known-good files: `dv/config.py`, `dv/data/{prep.py,dataset.py}`, `dv/model/{backbone.py,heads.py,build.py}`,
`train.py`, `scripts/{colab_setup.sh,sync.sh,rsh.sh,remote_bg.py,test_equivalence.py,test_numerics.py}`, `configs/v0_4b.json`.

Checks passed
- `scripts/test_equivalence.py` (tiny random Qwen3.5 config, fp32, CPU): shared-prefix forward == HF reference forward
  on prefix+block (max rel err ~1e-7), question blocks isolated from each other (causal and block_bidir).
- `scripts/test_numerics.py` (real Qwen/Qwen3.5-4B, GPU, fla kernels): ours-fp32 vs HF-fp32 cosine >= 0.99999;
  ours-bf16 deviation == HF-bf16's own deviation.
- Overfit: Qwen/Qwen3.5-4B, 50 records (~1.4k questions), 150 steps, 4 images/step: loss 10.1 -> 6.0,
  presence BCE 0.70 -> 1e-4, GIoU loss 0.85 -> 0.37, train Acc@IoU0.5 0.00 -> 0.94. Log: `outputs/v0/overfit/train_log.jsonl`.
  Peak memory 13.5 GB (RTX PRO 6000 96GB, bf16 base, LoRA r=64, grad ckpt).

Launch with a different model (on the VM, after `scripts/colab_setup.sh` and `scripts/sync.sh SESSION`):
```
cd /content/dv
python train.py --cfg configs/v0_4b.json --set model_id=Qwen/Qwen3.8-27B quant=nf4 lora_r=32 lora_alpha=64 \
    images_per_step=2 grad_accum=4 max_vis_tokens=640 out_dir=/content/out/v0_27b
# overfit smoke first:
python train.py --cfg configs/v0_4b.json --set model_id=... limit_records=50 max_steps=150 warmup_steps=10 grad_accum=1 out_dir=/content/out/overfit_27b
```
Layer types / hidden size / GDN dims are read from the HF config; `quant` in {bf16, nf4, fp8} (nf4/fp8 paths are
written but NOT yet tested). Validate a new model id with `python scripts/test_numerics.py <model_id>` first.
Env gotchas: Colab's preinstalled torchao 0.10 breaks peft LoRA injection -> `pip uninstall -y torchao` (setup script does it).
`colab exec` default 30 s silence timeout -> launch long jobs detached (`scripts/remote_bg.py`).
