# Base Qwen3.5-4B on ScreenSpot-Pro at native resolution

**Question.** An earlier comparison set Glance's ScreenSpot-Pro score of 50.7 against the generative base's 56.0, but the two numbers come from different input resolutions. Glance read each screenshot at native resolution (cap 16,384 visual tokens, no upscaling, median 4,080 tokens). The base was capped at 4,000 tokens. What does the base score when it reads the screenshots the same way Glance does?

**Answer.** It scores **59.9** on all 1,581 items. On the same items, the 4,000-token run scores 56.0. Native resolution helps the base by +3.9 points (paired bootstrap 95% CI [+2.0, +5.8]; exact McNemar p = 1e-4, with 149 vs 87 discordant items). At matched resolution, the gap between Glance and the base is therefore **9.2 points (50.7 vs 59.9)**, not the 5.3 points implied by 50.7 vs 56.0.

| Run | Items | Visual tokens (median) | Accuracy |
|---|---|---|---|
| Base, native (this run, `base_sspro_16384u1.jsonl`) | 1,581 | 4,080 | **59.9** |
| Base, cap 4,000 (`eval_v0_xdomain/rows/base_sspro_4000.jsonl`) | 1,581 | 3,948 | 56.0 |
| Glance v1.1 final, native (`sprint_gui` `v11f_sspro_16384u1`) | 1,581 | 4,080 | 50.7 |

The base run's median visual-token count is 4,080, exactly the same as Glance's run. Both runs use the same resize function, so the two models read identical pixel grids.

## Setting
- **Model:** `Qwen/Qwen3.5-4B`, bf16, HF `generate`, greedy, `enable_thinking=False`, at most 32 new tokens.
- **Prompt and scoring:** the same code path as `base_sspro_4000` (`evals/run_baseline.py --prompt click`). This is the official ScreenSpot-Pro Qwen3.5 `computer_use` prompt, with the assistant turn pre-filled up to `"coordinate": [`. A hit means the predicted point falls inside the ground-truth box.
- **Resolution:** `resize_dims(W, H, 16384, max_upscale=1.0)`, in multiples of 32 px, one visual token per 32×32 patch. This is the rule Glance's `v11f_sspro_16384u1` used (`evals/sprint/infer.py`). Inputs range from 2,040 to 15,698 tokens, and 942 of the 1,581 items exceed 4,000. The 4,000-token run also allowed small screenshots to be upscaled up to 2×.
- **Environment:** transformers 5.23.0 and flash-linear-attention 0.6.0, as in `evals/run_all.sh`, on one G4 (RTX PRO 6000). The run took 1,432 s, a median of 564 ms per item, and every output parsed.

## Where native resolution helps
| Slice | n | Cap 4,000 | Native |
|---|---|---|---|
| icon | 604 | 29.0 | 36.3 |
| text | 977 | 72.7 | 74.5 |
| OS | 196 | 49.0 | 60.7 |
| Dev | 299 | 52.5 | 60.2 |
| Creative | 341 | 52.8 | 55.7 |
| CAD | 261 | 49.4 | 50.6 |
| Office | 230 | 73.0 | 75.2 |
| Scientific | 254 | 61.0 | 60.2 |

Most of the gain is on icons (+7.3): small targets that get lost when 4K screenshots are shrunk to 4,000 tokens.

## Reconciling 56.0 with S1's "55.6"
The S1 report says the base 56.0 came from only 718 items and recomputes it as 55.6. That was true of the file S1 saw, but the file has since been completed, and **56.0 on 1,581 items is the correct number**.
- Commit `dc66d06` committed a partial `base_sspro_4000.jsonl` of 718 rows, with a mean hit rate of 55.57%, which rounds to 55.6. S1 recomputed from this partial file.
- Commit `f804073` completed the file to 1,581 rows, giving 55.98, which rounds to 56.0. Its first 718 rows match the partial file hit for hit, so the two numbers differ only in how many items they cover, not in scoring.
- The S1 branch does not contain `f804073`, and both reached `main` within minutes of each other. S1's caveat was therefore out of date, and the earlier "all 1,581 items … 56.0" matches the committed rows.

## Implication
The like-for-like ScreenSpot-Pro comparison is Glance 50.7 vs base 59.9, both reading native resolution in a single pass. The earlier sentence "at ≤4,000 visual tokens; the base reaches … 56.0" still correctly describes that run, but it puts the base 3.9 points lower than it scores in Glance's setting.

## Files
- `evals/run_baseline.py`: adds the `--shuffle_seed` and `--time_limit` options for seeded random-subset runs. Both are off by default, which keeps the old behaviour. This run used seed 0 and finished all items within the limit.
- `evals/sspro_base/run_vm.sh`: the VM driver, covering setup, data download and the run.
- `evals/sspro_base/analyze.py`: produces `results.json`, including the per-slice table and the reconciliation.
- `outputs/sspro_base/base_sspro_16384u1.jsonl`: per-item rows with id, hit, point, n_in_tokens, in_wh, ms and output text.
- **Cost:** one G4 on account `acct-8`, 17:32–18:06 UTC (34 min, about 5.7 CU at 10.05 CU/h). The VM is stopped, and nothing was written to HF.
