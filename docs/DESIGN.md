# Design notes (v0 and v1)

Moved from the monorepo README during the split. Commands assume a Colab VM with the repo at `/content/dv`.

## Shared-prefix forward (the efficiency trick)

One record = one image + many question blocks. The image prefix `<vision_start><image_pad>*n<vision_end>` is
run **once**; each question block `<object_ref_start>expr<object_ref_end><box_start>` continues from it
independently (`dv/model/backbone.py`):

- GDN layers: each block starts from the prefix's final recurrent state + last 3 conv inputs (causal, no
  leakage between blocks);
- full-attention layers: each block attends to its prefix + itself (bidirectional within the block with
  `attn_mask=block_bidir`, or causal);
- readout = final hidden state at `<box_start>`.

`scripts/test_equivalence.py` checks that this matches HF's reference forward on `prefix+block` (≈1e-7 in fp32,
tiny random config, CPU) and that blocks can't see each other; `scripts/test_numerics.py <model_id>` does the same
with real weights on GPU (fp32 cos ≥ 0.99999; bf16 deviation equals HF's own bf16 noise).

## Layout

```
dv/config.py            all knobs (JSON + --set key=value)
dv/data/prep.py         RefCOCO/+/g + gRefCOCO -> data/prep/{train,eval_*}.jsonl (run locally, CPU)
dv/data/dataset.py      image resize (<= max_vis_tokens), tokenization, collate (left-padded prefixes)
dv/model/backbone.py    shared-prefix forward over the hybrid LLM
dv/model/heads.py       presence / center / edges / IoU heads + losses (BCE, soft CE, Cramér, GIoU)
dv/model/build.py       load base (bf16|nf4|fp8), inject LoRA, save/load checkpoint (LoRA + heads only)
train.py  eval.py  baseline.py
scripts/colab_setup.sh  env + weights + COCO train2014 on a Colab VM
scripts/sync.sh         push code + data/prep to a pooled Colab session
scripts/rsh.sh          run a shell command on the VM; scripts/remote_bg.py launches detached jobs
scripts/profile_step.py torch.profiler over a few training steps
configs/v0_4b.json
```

## Run

```bash
# 1. annotations (local): HF jxu124/{refcoco,refcocoplus,refcocog}, FudanCVL/gRefCOCO, COCO instances_train2014
python dv/data/prep.py                 # -> data/prep/
# 2. Colab
colab-pool new dv-v0 --gpu A100
scripts/sync.sh dv-v0
#    on the VM: bash /content/dv/scripts/colab_setup.sh > /content/setup.log 2>&1
# 3. overfit test, then the real run
python train.py --cfg configs/v0_4b.json --set limit_records=50 max_steps=150 warmup_steps=10 out_dir=/content/out/overfit
python train.py --cfg configs/v0_4b.json
# 4. eval + baseline
python eval.py --ckpt /content/out/v0/ckpt_final.pt --latency --out /content/out/v0/eval_results.json
#   (default sets: RefCOCO/+ val/testA/testB, RefCOCOg val/test, grefcoco_val_merged)
python baseline.py --model Qwen/Qwen3.5-4B --n 500 --out /content/out/v0/baseline_results.json
```

Other model sizes: `--set model_id=<hf id> quant=nf4 lora_r=32 images_per_step=2 grad_accum=4 max_vis_tokens=...`.

## Data decisions (v0)

- gRefCOCO multi-target expressions are dropped (single-box head); single-target ones duplicate RefCOCO train
  and are de-duplicated; no-target ones are used as absent questions.
- Extra negatives: descriptive expressions (RefCOCOg, or >=3 words) from other images whose referent COCO
  category has no instance in the target image, up to a 25% absent ratio.
- Any image appearing in any eval split (RefCOCO/+/g val/test*, gRefCOCO val/testA/testB) is removed from
  training (RefCOCOg-umd train overlaps RefCOCO-unc val/test images).
- Images are resized so that the visual token count <= `max_vis_tokens` (640 by default; COCO 640x480 -> 29x22).

- This gRefCOCO release contains no single-target *val* expressions (val = 8,905 no-target + ~5.3k multi-target);
  its single-target val refs are exactly RefCOCO-unc val. `eval_grefcoco_val_merged` = gRefCOCO val + RefCOCO val
  (same 1,500 images) is used for N-acc / T-acc(single) / AUROC / ECE.

## Model / training decisions (v0)

- Center target: area-weighted over **all patches covered by the GT box** (GUI-Actor style, `center_target=box`);
  edges are trained conditioned on a patch sampled from that target, and decoded at the argmax patch.
- Block-bidirectional mask ON during training and inference (`attn_mask=block_bidir`); GDN layers causal.
- Losses: presence BCE (all), center soft-CE, Cramér (CDF-L2, Gaussian-smoothed target, sigma=1 bin) on 4 edges,
  GIoU on the expected box, BCE IoU head (target = IoU(decoded box)>=0.5). Absent questions: presence only.
- LoRA r=64/alpha=128 on all LLM linears (attention, GDN in/out projections, MLP), fp32 adapter weights under bf16
  autocast; frozen vision tower and embeddings; heads fp32. AdamW, cosine, warmup 100.
- causal-conv1d is not built: the custom forward uses torch depthwise conv (k=4). fla 0.6.0 provides the GDN kernel.
- Colab's preinstalled torchao 0.10 breaks peft LoRA injection; setup uninstalls it.

## v0 results (Qwen/Qwen3.5-4B, 1 epoch, RTX PRO 6000)

See `outputs/v0/` (local, git-ignored): `eval_results.json`, `baseline_results.json`, `ours_on_baseline_subsets.json`,
`run_summary.json`; checkpoints in `checkpoints/v0/`. `python scripts/make_table.py outputs/v0/eval_results.json outputs/v0/baseline_results.json`
prints the table.

## v1: general typed decision model (choice / score / noul + box)

Same shared-prefix forward; the state (image and/or text) is the prefix, every question is a block that continues it
(`dv/data/mixed.py` documents the chat-style prompt layout). Readout at the block's last token:

- `box`: v0 heads (presence, patch center, 4 edge distributions, IoU head); point-only questions (PixMo, AGUVIS clicks)
  train presence + center only.
- `choice` / `score` / `noul`: softmax over the block's label tokens (A..Z, or {No,Yes}); logits = LM-head rows of those
  tokens (= the base model's next-token logits at init) + a zero-init linear residual (`dv/model/typed.py`).
  Losses: soft-target CE + Brier (choice, noul), CE + squared Cramér over levels (score).
- Post-hoc temperatures (presence, choice, noul, score) are fitted by `eval.py` on held-out training-distribution splits
  (`calib_*.jsonl`), never on eval sets.

```bash
# VM: raw data, pools, mixture
bash scripts/v1/fetch.sh > /content/fetch.log 2>&1
python -m dv.data.prep_v1 natural   # (local OK: annotations only)
bash scripts/v1/prep_all.sh         # text / evalhash / gui / pixmo pools
python -m dv.data.prep_v1 postdedup --out /content/data/prep_v1
python -m dv.data.prep_v1 mix --out /content/data/prep_v1 --total N
python scripts/v1/test_v1_init.py --image <coco jpg>     # typed head at init == HF logits on the label tokens
python train.py --cfg configs/v1_4b.json
python eval.py --ckpt /content/out/v1/ckpt_final.pt --calib_dir /content/data/prep_v1 \
   --sets refcoco_val,...,grefcoco_val_merged,opendecisions,ss2,sspro,venus --latency --out results.json
python eval.py --base --sets opendecisions ...          # zero-shot base (typed head = LM readout)
```
`configs/v0_4b.json` still runs the v0 path unchanged (`task` defaults to `v0`).
