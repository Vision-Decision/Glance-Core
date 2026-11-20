# Output-head ablation: run plan (estimated before launch)

VM: one Colab G4 (RTX PRO 6000 96GB) `dv-abl`, ~10 CU/h.
Fixed for every run: Qwen/Qwen3.5-4B bf16, LoRA r64/a128 on all LLM linears (vision/embeddings frozen),
max_vis_tokens 640, 8 images/step, grad_accum 1, 3058 steps (= 1 epoch of train.jsonl), AdamW lr 2e-4 (LoRA),
cosine/warmup 100, seed 0, dataloader generator seed 0 (identical image order and question subsampling in every run).
Only the output interface changes (see ablation/common.py docstring).

| # | run | variant | train data | est. train h | est. eval h |
|---|---|---|---|---|---|
| 1 | ours_neg25 | presence gate + patch + edge ordinal (v0 head, retrained in this harness) | 25% absent | 1.3 | 0.1 |
| 2 | ar_digits_neg25 | "x1,y1,x2,y2" 0-1000 digit tokens / "none" | 25% | 1.5 | 0.4 |
| 3 | ar_hier_neg25 | 2-level coarse/fine new tokens (32x32 = 1024 levels) | 25% | 1.5 | 0.25 |
| 4 | ar_atomic1000_neg25 | 1 new token per value, 1000 bins | 25% | 1.5 | 0.2 |
| 5 | ar_atomic100_neg25 | 1 new token per value, 100 bins | 25% | 1.5 | 0.2 |
| 6 | ours_neg05 | ours | 5% absent (absent subsampled, positives identical) | 1.2 | 0.1 |
| 7 | ar_digits_neg05 | AR-digits | 5% | 1.4 | 0.4 |

Plus: env/COCO setup ~0.25 h, sanity + throughput benchmark ~0.3 h, latency for all runs ~0.3 h.
Total estimate ≈ 12-13 GPU-hours (≈ 125 CU) sequential; below the 15 GPU-hour approval threshold.
If two jobs on the one GPU give >1.3x aggregate throughput, runs are paired to cut VM-hours.

Eval sets: gRefCOCO val merged (= RefCOCO val 10,834 present + 8,905 no-target + 5,324 multi-target),
RefCOCO testA, testB. Per-question rows are saved for the entropy analyses.
