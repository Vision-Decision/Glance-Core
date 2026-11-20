# Sources for the Glance-4B-preview benchmark table

Glance-4B-preview is the v1 checkpoint (`glance_config.json`: `"version": "v1"`, `"train_step": 3844`). Every
Glance cell is that checkpoint. Paths are in glance-core unless another repository is named.

| Row | Glance-4B-preview | Qwen3.5-4B | Qwen3-VL-4B | Qwen3-VL-8B | Glance-X-4B joint |
|---|---|---|---|---|---|
| RefCOCO val | 93.17: `outputs/v1/eval_results.json` `sets.refcoco_val.acc50` | 91.4 (500-item random sample, generative, greedy): `outputs/v0/baseline_results.json` `sets.refcoco_val.acc50` | — | — | 93.1†: Glanece-Training `docs/RESULTS.md` §1, `joint` (our image_first reproduction, 92.86, is in `outputs/glancex/summary.md`) |
| RefCOCO testA | 94.52: same file, `refcoco_testA` | — | — | — | 94.9†: same |
| RefCOCO testB | 90.66: `refcoco_testB` | — | — | — | 90.5†: same |
| RefCOCO+ val / testA / testB | 89.67 / 93.33 / 84.58: `refcocoplus_*` | — | — | — | 89.6 / 93.1 / 84.9†: Glanece-Training `docs/RESULTS.md` §1, `joint` |
| RefCOCOg val / test | 89.95 / 90.44: `refcocog_*` | — | — | — | 90.3 / 90.5†: same |
| 8-split mean | 90.79: mean of the 8 `acc50` values | 88.1†: Qwen3.5-4B model card, "RefCOCO(avg)" (splits not stated) | — | — (the 89.1 on the earlier card could not be located in arXiv 2511.21631; dropped) | 90.9†: Glanece-Training `docs/RESULTS.md` §1, `joint` mean |
| gRefCOCO val, no target rejected | 87.40: `sets.grefcoco_val_merged.N_acc` (P(present) < 0.5) | 65.0 (500-item sample, answered "none"): `outputs/v0/baseline_results.json` `grefcoco_val_notarget.N_acc` | 77.2†: RC-GRPO (arXiv 2608.04698) Table 1, Qwen3-VL-4B-Instruct val N-acc; copied in `outputs/sprint_abs/literature.md` | — | 78.9†: Glanece-Training `docs/RESULTS.md` §1, `joint`, val (merged) `reject_rate` |
| gRefCOCO val, AUROC of P(present) | 97.92: `grefcoco_val_merged.auroc_presence` | — | — | — | 98.2†: Glanece-Training `docs/RESULTS.md` §1, `joint` |
| gRefCOCO testA / testB, no target rejected | 80.8 / 77.1: `outputs/sprint_abs/v1/tables.md`, grec_testA / grec_testB single_raw@0.5, N-acc | — | 70.2 / 70.2†: RC-GRPO Table 1 (Qwen3-VL-4B-Instruct N-acc) via `literature.md` | — | 71.3 / 73.0†: Glanece-Training `docs/RESULTS.md` §1, `joint` `reject_rate` |
| gRefCOCO testA / testB, AUROC of P(present) | 94.1 / 88.7: `outputs/sprint_abs/v1/tables.md`, presence calibration section, raw | — | — | — | 97.4 / 93.5†: Glanece-Training `docs/RESULTS.md` §1, `joint` |
| GREC testA / testB Pr@(F1=1) (official) | 47.5 / 51.3: `outputs/sprint_abs/v1/tables.md`, single_raw@0.7(official) | — | 68.0 / 62.2†: RC-GRPO Table 1 via `literature.md` | — | — |
| gRefCOCO val, GREC Pr@(F1=1) (official, threshold 0.7) | 71.6: `outputs/sprint_abs/v1/tables.md`, grec_val single_raw@0.7(official) | — | 72.4†: RC-GRPO Table 1 via `outputs/sprint_abs/literature.md` | — | — |
| VenusBench-GD refusal (official UI-Venus scorer) | 81.76: `outputs/sprint_abs/v1/tables.md`, raw_gated, Refusal | — | 31.55†: VenusBench-GD (arXiv 2512.16501) Table 5 via `literature.md` | 63.78†: same | — |
| gRefCOCO val multi-target F1 | — (v1 returns one box per question) | — | — | — | 82.7†: Glanece-Training `docs/RESULTS.md` §1, `joint` `multi_f1` |
| ScreenSpot-v2 click accuracy | 89.39 (≤2,048 visual tokens, upscale ≤2): `outputs/v1/eval_results.json` `sets.ss2.raw.click_acc` | 93.4 (generative, ≤4,000 visual tokens): `outputs/sprint_gui/summary.md`; rows `outputs/eval_v0_xdomain/rows/base_ss2_4000.jsonl` | — | 92.1†: UI-Venus-1.5 report (arXiv 2602.09082) Table 1 (its reproduction) | — |
| ScreenSpot-Pro, native resolution | 39.34: `outputs/sprint_gui/raw/v1_sspro_16384u1.info.json` `click_acc_box` | 59.9: `outputs/sspro_base/summary.md` (same resize rule, native) | — | — (the 54.6 on the earlier card could not be located in arXiv 2511.21631; dropped) | — |
| ScreenSpot-Pro, zoom track (2 passes) | 58.13: `outputs/sprint_gui/raw/v1_sspro_16384u1_zoom0.25_4096u2box.info.json` | — | — | — | — |
| VenusBench-GD overall (official) | 40.06: `outputs/sprint_abs/v1/tables.md`, raw_gated, Overall | — | 56.15†: VenusBench-GD Table 5 | 62.34†: same | — |
| 8 box questions, one image (ms) | 85.6: `outputs/v1/eval_results.json` `latency.box_8q.ms_median` (85.63) | 3,161.2: `outputs/v0/baseline_results.json` `latency_ms_median_8q_same_image` (8 sequential HF `generate` calls) | — | — | — |
| 32 box questions, served (ms) | 48.9: glance-engine `outputs/engine_opt/CEILING.md`, final-state table, one image box Q=32 (same v1 weights) | 334.0: same row, vLLM BF16 serving Qwen3.5-4B | — | — | — |

† reported by others, with their own prompts and resolutions. Cells are coloured by comparing the values as shown
(one decimal, two for VenusBench-GD); ties colour every tied cell.
