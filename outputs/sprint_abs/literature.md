# Literature numbers used for comparison (checked against arXiv HTML / official code, 2026-11-18)

## GREC official protocol (gRefCOCO; GREx arXiv 2601.05244 §3.3, GREC tech report 2308.16182, code henghuiding/gRefCOCO mdetr/datasets/refexp.py)
- Prediction set = every predicted box with score >= 0.7 (`thresh_score=0.7`; 2308.16182 Tab. 3 caption "Threshold is set to 0.7 for all the methods").
- Greedy one-to-one matching on **GIoU** >= 0.5 (code uses `generalized_box_iou`; the paper text says IoU).
- Per-sample F1 = 2TP/(2TP+FP+FN); success iff F1 = 1; no-target sample succeeds iff no box survives.
- Pr@(F1=1, IoU>=0.5) over all samples; N-acc over no-target samples; T-acc = target samples with a non-empty prediction.

| Method | val Pr/N-acc/T-acc | testA | testB | Source |
|---|---|---|---|---|
| MCN | 28.02 / 30.64 / 99.62 | 32.29 / 32.04 / 99.56 | 26.76 / 30.27 / 99.80 | GREx Tab. 14 |
| VLT | 36.62 / 35.20 / 99.44 | 40.21 / 34.07 / 99.39 | 30.24 / 32.53 / 99.56 | GREx Tab. 14 |
| MDETR | 42.69 / 36.27 / 99.40 | 50.04 / 34.49 / 99.99 | 36.52 / 31.02 / 99.63 | GREx Tab. 14 |
| UNINEXT-R50 | 58.19 / 50.58 / 96.52 | 46.41 / 49.33 / 96.87 | 42.91 / 48.22 / 98.16 | GREx Tab. 14 |
| ReLA (Swin-B) | 61.90 / 56.37 / 96.32 | 50.35 / 59.02 / 97.68 | 44.61 / 58.40 / 95.89 | GREx Tab. 14 |

MLLMs (RC-GRPO arXiv 2608.04698 Tab. 1; Pr@(F1=1,IoU>=0.5) / N-acc; multi-target handling for MLLMs not stated):

| Model | val | testA | testB |
|---|---|---|---|
| RC-GRPO-II (Qwen3-VL-4B) | 75.7 / 85.3 | 71.4 / 83.1 | 64.8 / 78.4 |
| RC-GRPO-I (Qwen3-VL-4B) | 74.4 / 84.7 | 69.0 / 81.3 | 63.2 / 77.5 |
| Qwen3-VL-4B-Instruct | 72.4 / 77.2 | 68.0 / 70.2 | 62.2 / 70.2 |
| Qwen3-VL-235B-A22B | 59.5 / 65.7 | 58.3 / 66.4 | 52.3 / 56.5 |
| Qwen2.5-VL-7B | 38.2 / 6.4 | 47.2 / 3.4 | 38.3 / 4.2 |
| CRS (Qwen3-4B) | 56.3 / 89.5 | 46.4 / 87.3 | 47.5 / 84.0 |
| InstanceVG | 73.5 / 72.8 | 70.2 / 71.1 | 60.8 / 65.2 |
| HieA2G-R101 | 67.8 / 60.3 | 66.0 / 60.1 | 56.5 / 56.0 |

VGE (arXiv 2611.22625) reports only **mean per-sample F1** (not Pr@F1=1), testA/testB, plain IoU, threshold tuned on its own
train-split val sample: VGE 80.6 (testA) / 72.8 (testB); by target count (None/Single/Multi, mean of 3 instruction variants)
testA 86.0/84.0/76.5, testB 75.7/73.0/71.0; Qwen3.5-9B zero-shot None-F1 27.2 (testA) / 14.8 (testB). Latency 139.9 ms mean.

## VenusBench-GD (arXiv 2512.16501 Tab. 5; UI-Venus-1.5 arXiv 2602.09082 Tab. 9). Overall = micro-average over 6,166 items incl. 900 refusal.

| Model | Basic avg | Reasoning | Functional | Refusal | Advanced avg | Overall | Source |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B | 72.54 | 25.38 | 55.43 | 31.55 | 35.20 | 56.15 | VenusBench Tab. 5 |
| Qwen3-VL-8B | 76.96 | 24.75 | 47.71 | 63.78 | 43.66 | 62.34 | VenusBench Tab. 5 |
| Qwen2.5-VL-72B | 75.83 | 31.44 | 50.14 | 78.11 | 51.79 | 65.28 | VenusBench Tab. 5 |
| UI-Venus-Ground-7B | 68.66 | 24.39 | 53.85 | 0.00 | 23.90 | 49.01 | VenusBench Tab. 5 |
| UI-Venus-Ground-72B | 83.12 | 46.16 | 68.86 | 51.33 | 53.75 | 70.23 | VenusBench Tab. 5 |
| GTA1-7B / UI-TARS-7B | 64.87 / 58.11 | | | 0.00 / 0.00 | | 46.38 / 39.65 | VenusBench Tab. 5 |
| UI-Venus-1.5-2B | 81.4 | 22.0 | 57.3 | 76.3 | 49.2 | 67.3 | UI-Venus-1.5 Tab. 9 |
| UI-Venus-1.5-8B | 86.6 | 38.1 | 70.1 | 61.6 | 54.2 | 72.3 | UI-Venus-1.5 Tab. 9 |
| Qwen3-VL-8B* (their repro) | 76.8 | 22.6 | 61.3 | 6.8 | 27.3 | 55.1 | UI-Venus-1.5 Tab. 9 |

The HF release of `refusal_grounding.json` has **899** items (6,165 total), one fewer than the paper's 900.
