# Gen-v1: Glance v1 trained to answer in text, on the same data
Gen-v1 = Glance v1's backbone (Qwen3.5-4B), LoRA (r=64/α=128, all LM linears, frozen vision), data recipe, prompts, shared-prefix batching, schedule and epoch, with every answer produced as tokens with CE: box -> "x1,y1,x2,y2" (0–1000 digits) or the single token "none"; point -> "x,y" (first point); choice/yes-no/score -> label token. Read-outs: first-token dial q(none), greedy fixed point, grammar-constrained greedy box, and a post-hoc IoU head (ablation/ar_iou_head.py recipe) trained on held-out images.

Bottom line: on RefCOCO/gRefCOCO the data explains almost all of Glance v1's grounding and absence numbers (box accuracy tie, N-acc slightly better for Gen greedy, same raw presence ECE). The interface still wins on box confidence (κ̂ +1.9 AUROC), presence ranking (+0.4 AUROC) and two-knob cost: 4–9% cheaper at low c_W; with box accuracy held equal, cheaper at every c_W under P2 and at c_W ≤ 1 under P1 (tie at c_W = 5). On GUI, the generative "none" token over-refuses (SS-v2 false refusal 43% vs 16%; official VenusBench-GD 28.2 vs 40.1), while ungated clicks are equal or better.

Source: outputs/gen_v1/results.json (gen_v1/analyze.py). CIs: 95% image bootstrap (1000), paired for Δ. Glance rows: HF Vision-Decision/glance-train-ckpts-legacy sprint_abs/rows_v1/. Rows joined on (image, expression, presence, GT box); all joins complete.

## gRefCOCO val merged (25,063; 8,905 no-target; 10,834 single; 5,324 multi) — key prefix grefcoco_val_merged.
| metric | Glance v1 | Gen-v1 | Δ (G−Gen) | key |
|---|---|---|---|---|
| Acc@0.5 RefCOCO val (forced box) | 93.19 [92.50,93.89] | 93.59 [92.85,94.29] | −0.41 [−1.01,+0.21] | box_acc50_single_target |
| N-acc (ĝ<.5 / greedy none) | 87.36 [86.65,88.04] | 88.37 [87.71,89.01] | −1.01 [−1.60,−0.46] | N_acc.* |
| N-acc Gen q(none)>.5 | – | 86.48 [85.75,87.16] | | N_acc.gen_pnone_0_5 |
| false refusal single / multi-target | 1.37 / 5.11 | 1.93 / 8.66 | | refusal_single_target, refusal_multi_target |
| presence AUROC | 97.92 [97.74,98.12] | 97.48 [97.28,97.69] | +0.44 [+0.34,+0.56] | presence.* |
| presence ECE raw / cross-fit Platt | 2.27 / 0.65 | 2.26 / 0.76 | | presence.ece_* |
| κ AUROC (κ̂ / IoU head) | 90.19 [88.69,91.68] | 88.30 [86.54,90.00] | +1.89 [+0.62,+3.23] | kappa_present_single_target.* |
| κ ECE raw | 0.56 | 0.79 | | kappa_present_single_target.ece_raw_* |
| Gen κ AUROC: scalar head / P(ŝ) / len-norm | | 79.69 / 69.46 / 64.34 | | kappa_present_single_target.* |
IoU head: trained on H = 12,888 RefCOCOg val+test expressions on images outside gRefCOCO val/testA/testB (unseen in training); val AUROC 87.6 (head_scores_info.heldout_full). Other REC splits not decoded (budget).

## Two-knob ×100, c_H=c_R=1 (twoknob.costs_cH=cR=1.<protocol>.cW=<c>), test n=19,739, C n=20,669
| c_W | P1 G / Gen / Gen+head | P1 Δ | P2-known G / Gen+head | P2-known Δ | P2-EM Δ |
|---|---|---|---|---|---|
| 0 | 5.06/5.54/5.54 | −0.48 [−0.77,−0.20] | 5.35/5.77 | −0.42 [−0.69,−0.15] | −0.41 [−0.68,−0.14] |
| 0.5 | 6.65/7.04/6.94 | −0.28 [−0.58,+0.04] | 6.78/7.23 | −0.45 [−0.75,−0.15] | −0.47 [−0.78,−0.17] |
| 1 | 8.19/8.45/8.30 | −0.11 [−0.49,+0.28] | 8.34/8.68 | −0.34 [−0.72,+0.06] | −0.32 [−0.72,+0.08] |
| 2 | 10.82/11.16/10.81 | +0.01 [−0.56,+0.60] | 11.14/11.56 | −0.42 [−1.00,+0.13] | −0.40 [−0.98,+0.15] |
| 5 | 15.70/18.96/15.65 | +0.05 [−0.97,+1.14] | 15.58/16.24 | −0.66 [−1.61,+0.24] | −0.63 [−1.56,+0.28] |
Box error test 6.8% (G) / 6.4% (Gen); on C 12.0% / 10.8% (twoknob.test_box_error, cal_box_error). Priors in twoknob.priors.
Box-equalised (twoknob.boxeq; 19,177 items, 562 present dropped): P1 Δ −0.46, −0.37, −0.29, −0.26, +0.01; P2-known Δ −0.33, −0.50, −0.45, −0.56, −0.84 (all CIs exclude 0); P2-EM similar.

## GUI (gui.*, venus_official.*)
SS-v2 (1,272): click ungated 89.23 vs 87.81; gated 77.04 vs 52.67; false refusal 16.4 vs 43.1.
VenusBench-GD: click ungated 47.28 vs 50.97; refusal acc 81.76 vs 95.77; false refusal 46.5 vs 76.8; presence AUROC 73.8 vs 74.2. Official overall: Glance 40.06 (reproduces sprint_abs), Gen greedy 28.18, Gen q(none)>.5 29.80.

## Training / read-outs
Data rebuilt with 942e648 code + v1's own natural pool: 212,297 records / 520,027 questions (v1: 212,373 / 520,048); HF baselines/gen_v1/data/. 3,850 steps, 391.4 min, 6.09 s/step, 27.7 GB peak; final box CE 0.59, typed CE 0.23; q(none) on training batches 0.018 present / 0.916 absent (train_log.jsonl). Prefix-LM mask in each block (question tokens bidirectional as in v1's block_bidir, answers causal). Incremental decoder checked against ablation ar_decode on the first batch of every set: box IoU .97–.996, median |Δq(none)| ≤ 4e-4; bf16 floor (ar_decode vs itself re-batched): 22% exact tokens, IoU .98.

## Not identical to v1
PixMo URL downloads (25,524 vs 24,006 of 50k) and ~0.1% text/GUI pool drift, so the sampled records and order differ (totals and per-type shares match within ~1%). Prefix-LM mask for answer tokens. "Yes"/"No" noul tokens. First point only on point items. Only RefCOCO val of the 8 REC splits; no in-sample IoU head; no latency.

## Compute
~104 CU (one G4, 17:48–04:10 UTC; 6.5 h training, ~40 min decoding, ~2.8 h idle after the first dump aborted). Data prep and analysis on gen-host (CPU). VM stopped.
