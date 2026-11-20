---
license: other
license_name: research-non-commercial
license_link: LICENSE
base_model: Qwen/Qwen3.5-4B
library_name: peft
pipeline_tag: image-text-to-text
tags:
- decision-model
- grounding
- gui-grounding
- referring-expression-comprehension
- calibration
- system-one
---

# decision-vision v1.1 (4B)

A **single-prefill decision model** on top of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B): one forward
pass over a state (an image, a text, or both) answers many typed questions at once and returns **calibrated
probabilities** instead of generated text. There is no autoregressive decoding.

| Primitive | Question | Output |
|---|---|---|
| `box` | a referring expression / GUI instruction | `P(present)`; a box (center patch + 4 ordinal edge distributions); `P(IoU≥0.5)`; confidence = `P(present)·P(IoU≥0.5)` |
| `choice` | instruction + N options (N ≤ 26) | probability per option |
| `score` | instruction + K ordered levels | probability per level (ordinal) |
| `noul` | yes/no question | `P(yes)` |

`P(present)` lets the model say "the target is not there" (refusal) with a calibrated confidence. The model answers
the same question types as Jev / OpenDecisions-style typed decision models, and adds the `box` primitive.

**This is a research release under a non-commercial license (see [License](#license)).**

## How it works

- **Shared prefix.** The state is prefilled once as `<|im_start|>user\n[<vision_start><image_pad>*n<vision_end>][state text]`.
  Each question is a short block that continues that prefix:
  - in the 24 Gated-DeltaNet layers, a block starts from the prefix's recurrent state and conv tail;
  - in the 8 full-attention layers, a block attends to the prefix plus itself (bidirectional inside the block).

  Blocks never see each other, so 1 question and 8 questions cost about the same.
- **Readout.** Every question is read out at the last token of its block.
  - `box`: presence gate, a softmax over visual tokens (32×32 px patches) for the center, 4 × 64-bin edge offset
    distributions, and an IoU head.
  - Typed questions: a softmax over the block's label tokens (`A..Z`, or `{No, Yes}`). The logits are the LM-head rows
    of those tokens plus a zero-initialized residual, so at initialization the readout equals the base model's
    next-token distribution.
- **Losses.**
  - choice / noul: soft-target CE + Brier, with label smoothing 0.05 on one-hot choice targets.
  - score: CE + squared Cramér.
  - box: presence BCE, center soft-CE, Cramér on the edges, GIoU, IoU-head BCE.
- **Training setup.** LoRA r=64 on all LLM linears; frozen vision tower; one epoch.
- **Temperatures.** Post-hoc temperatures (presence, choice, noul, score) were fitted on held-out
  training-distribution splits, never on evaluation sets. The choice temperature comes from a held-out
  DecisionBank split restricted to the judge/hard difficulty tiers.

## Usage

```python
from inference import DecisionVision          # inference.py + the bundled `dv/` package from this repo
m = DecisionVision.from_pretrained(".")        # downloads Qwen/Qwen3.5-4B, applies adapter + heads
m.decide(image="screen.png", max_vis_tokens=6400, questions=[
    {"type": "box", "text": "the Save button in the toolbar"},
    {"type": "noul", "instr": "Is a modal dialog open?"}])
# -> [{"type": "box", "p_present": 0.97, "box": [x1, y1, x2, y2], "p_iou50": 0.88, "confidence": 0.85},
#     {"type": "noul", "p_yes": 0.03, "probs": [0.97, 0.03]}]
m.decide(text="CRM note: ...", questions=[
    {"type": "choice", "instr": "What will the customer order?", "options": ["less", "same", "more", "not mentioned"]},
    {"type": "score", "instr": "How urgent is it?", "options": ["low", "medium", "high"]}])
```

**Requirements:** `torch`, `transformers>=5.9`, `peft`, `flash-linear-attention==0.6.0`, `safetensors`, `pillow`.

**Files:**
- `adapter/` — LoRA, PEFT naming.
- `heads.safetensors` — box and typed heads.
- `dv_config.json` — config, label tokens and temperatures.
- `merged-bf16/` — LoRA merged into the base; load it with `from_pretrained(".", merged=True)`.

**Image resolution:**
- natural images: ≤ 640 visual tokens (the default);
- GUI screenshots: native resolution up to 6400–8192 visual tokens.

## Results

{TABLES}

### Public reference points (from the cited sources, not re-run by us)

| Benchmark | Model | Score | Source |
|---|---|---|---|
| RefCOCO val/testA/testB · RefCOCO+ val/testA/testB · RefCOCOg val/test | Florence-2-L (0.77B, fine-tuned generalist) | 93.4/95.3/92.0 · 88.3/92.9/83.6 · 91.2/91.7 | arXiv 2311.06242, Tab. 6 |
|  | InternVL3-8B | 92.5/94.6/88.0 · 88.2/92.5/81.8 · 89.6/90.0 | arXiv 2504.10479, Tab. 6 |
|  | Qwen2.5-VL-7B | 90.0/92.5/85.4 · 84.2/89.1/76.9 · 87.2/87.2 | arXiv 2502.13923, Tab. 6 |
|  | Qwen3.5-4B (base) | RefCOCO avg 88.1 | Qwen/Qwen3.5-4B model card |
| gRefCOCO val N-acc | InstAlign (Swin-B) | 79.72 | arXiv 2411.15087v2, Tab. 1 |
|  | GSVA-Vicuna-7B | 62.43 | arXiv 2312.10103, Tab. 1 |
| ScreenSpot-v2 | MAI-UI-8B / Step-GUI-8B / UI-Venus-Ground-7B | 95.2 / 95.1 / 94.1 | arXiv 2512.22047 Tab. 10; 2512.15431v2 Tab. 3; 2508.10833 Tab. 5 |
| ScreenSpot-Pro | MAI-UI-8B (zoom-in) | 65.8 (70.9) | arXiv 2512.22047, Tab. 2 |
|  | Step-GUI-8B / Qwen3.5-4B (thinking) / Qwen3-VL-4B | 62.6 / 60.3 / 59.5 | 2512.15431v2 Tab. 3; Qwen3.5-4B card; 2511.21631v2 Tab. 4 |
| VenusBench-GD basic avg / refusal / overall | Qwen3-VL-8B | 76.96 / 63.78 / 62.34 | arXiv 2512.16501, Tab. 5 |
|  | Qwen3-VL-4B | 72.54 / 31.55 / 56.15 | same |
|  | GUI-G2-7B · Holo1.5-7B | 72.02 / 0.00 / 50.46 · 70.54 / 0.00 / 51.00 | same |
| OpenDecisions Capability (single pass, 4B) | Torchcast-decision-4B v1.1 / Plumb-4B / Jeff-1.0-Qwen-4B | 73.5 / 73.1 / 67.7 | jgeuter/OpenDecisions card, leaderboard 2026-10-15 |
|  | Wald-4B v2.0 (reasons on 557 items) | 77.1 | same |

**Caveats on comparability:**
- Our VenusBench-GD "grounding" numbers count positive items only. The benchmark's official Overall also includes the refusal
  items; we report that separately with the official scorer.
- Published numbers come from generative models prompted to output points.

## Evaluation protocol

**Grounding**
- **RefCOCO family:** Acc@IoU≥0.5 of the decoded box, ignoring presence; this is standard REC.
- **gRefCOCO val:** merged with RefCOCO val, the same 1,500 images, because this release has no single-target val
  expressions.
  - N-acc = no-target expressions with `P(present)<0.5`.
  - AUROC and 15-bin ECE of `P(present)`.
  - Box-confidence ECE against the event "present and IoU≥0.5".
- **GUI benchmarks:** the click point is the center of the predicted box, scored point-in-GT-box.
  - Screenshots run at native resolution, capped at 8,192 visual tokens (1 token = 32×32 px).
  - Crop-and-zoom (SS-Pro, reported separately): a window of 40% of each side, centered on the first box, is
    re-prefilled at up to 4,096 tokens and the box is mapped back.
  - VenusBench refusal: correct iff `P(present)<0.5`.
  - The official VenusBench-GD scorer (`inclusionAI/UI-Venus@VenusBench-GD/eval.py`) is also reported.
- **OpenDecisions:** the official `score.py` (Capability = mean of Intelligence and Calibration).
  - Prompt: our typed-block layout.
  - The temperatures were never fitted on OpenDecisions.
- **Latency:** CUDA-synchronized model time, batch 1, on the GPU stated in the table.

## Training data (v1.1 mixture, 810k questions)

| Part | Questions | Sources | License |
|---|---|---|---|
| Typed text decisions | 202.5k | tasksource/tasksource-jev-typed-decisions (rows with `license_use != non-commercial`, train split)<br>jgeuter/DecisionBank-260k (sources whose stated license is NC or research-only removed)<br>inclusionAI/LemonDecider-Decision-Mix (per-row `license_use != non-commercial`) | mixed; some sources "unknown" / "other" |
| Natural-image grounding | 324k (28% absent; 65% of absents hard) | RefCOCO / RefCOCO+ / RefCOCOg (train), gRefCOCO no-target (train)<br>hard negatives: color swaps, same-category descriptive swaps | COCO images CC-BY; **gRefCOCO non-commercial** |
| GUI grounding | 227k (10% absent) | GroundCUA (MIT)<br>OS-Atlas desktop Windows/Linux/macOS and mobile (Apache-2.0)<br>AGUVIS stage-1 guienv (Apache-2.0) | MIT / Apache-2.0 (screenshots carry third-party UI content) |
| Pointing / counting | 57k | allenai/pixmo-points, pixmo-count (counts as `score`) | ODC-BY |

**Decontamination**
- Every image of every reported eval set was removed by COCO id or dHash (Hamming ≤3):
  - RefCOCO family and gRefCOCO val/test;
  - ScreenSpot-v2 and ScreenSpot-Pro;
  - VenusBench-GD;
  - PixMo-count val/test (by URL and sha);
  - Point-Bench.
- Text rows sharing any 10-word shingle with OpenDecisions or with the eval splits of Decision-Index public
  benchmarks were dropped. Those benchmarks are BANKING77, CLINC150, ARC, HellaSwag, WinoGrande, OpenBookQA, CSQA,
  ANLI, MMLU, MMLU-Pro and GSM8K.
- Sources with no official train split, and Decision-Index-specific sources, were excluded by name.
- **Note:** DecisionBank's `j11_generator` part, about 1% of the text, comes from the same generator code as
  OpenDecisions' programmatic items (different seed; no identical items, per its card).

## Limitations

- **Single box per question.** Multi-target expressions get one box, and the presence gate still fires for them.
- **Natural-image resolution.** Images run at ≤640 visual tokens by default, so very small objects in natural images
  are hard.
- **GUI screenshots.** GUI quality depends on native-resolution input. Training data is mostly English, and ~2/3 of
  it is desktop.
- **Calibration under shift.** Typed-decision calibration is fitted on in-distribution held-out data and can drift on
  new domains. Refit the temperatures on your own data.
- **Not for safety-critical use.** Outputs are probabilities and may still be overconfident under distribution shift.

## License

The weights are released for **research and non-commercial use only**:
- gRefCOCO is licensed non-commercial;
- several DecisionBank and LemonDecider upstream sources state "unknown" or "other" terms.

The base model's license (Qwen3.5-4B, Apache-2.0) also applies to the base weights contained in `merged-bf16/`.

## Citation

If you use this model, please also cite the datasets above and the base model.
