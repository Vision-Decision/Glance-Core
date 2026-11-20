# AR-digits + its own trained IoU head (an additional baseline)

Written from the experiment agent's report; every number is in `results.json` (keys below). Code: `ablation/ar_iou_prep.py`,
`ar_iou_feats.py`, `ar_iou_head.py`, `v3_ar_iou.sh`, `twoknob_v3.py`. Rows/features: HF `Vision-Decision/glance-train-ckpts`
`baselines/ar_iou/` (ar_iou_rows.tgz, ar_iou_feats/). One G4, ~58 min (~5–6 CU), VM stopped.

## Design
- Frozen `ar_digits_neg25`; grammar-constrained greedy re-decode reproduces `rows_c.json` exactly (100% same boxes; `build_check.json`).
  AR presence unchanged (Platt on margin to "none"); only κ replaced.
- Head = Glance's IoU-head shape/loss (LayerNorm→Linear 512→GELU→Linear 1; BCE on 1[IoU≥0.5]; early stopping). Inputs: final hidden
  states at <|box_start|>, mean over coordinate tokens, <|box_end|>; 16 scalars (token log-probs/margins, first-token margin to none,
  box geometry); 64 Fourier features of the box centre.
- grefcalib / calib_natural are NOT usable: all their images (600/600, 378/378) and positive expressions are v0 training data.
- Splits (pairwise image-disjoint from the test set): H (head training) 12,888 RefCOCOg val+test expressions on 3,549 images outside
  gRefCOCO val/testA/testB; C (calibration) gRefCOCO testA+testB single-target + no-target, 20,669 items (9,116 absent);
  T (in-sample variant) 10,815 present expressions on 1,000 v0 training images.

## Box quality on the 10,834 present test items (`box_quality_present_test.<key>.{auroc,ece_raw,risk_at_80cov}`)

| κ | key | AUROC | ECE raw | risk@80% |
|---|---|---|---|---|
| Glance κ̂ | glance_kappa_hat | 91.6 | 1.31 | 0.88 |
| AR box log-prob | ar_logprob_P(s_hat) | 70.9 | – | 3.76 |
| AR length-norm. conf (Tab. 2) | ar_lenorm_conf | 66.0 | – | 4.12 |
| AR + IoU head (H) | ar_head_heldout_full | 88.9 | 0.64 | 1.38 |
| AR + head in-sample (T) | ar_head_insample_full | 87.6 | 2.11 | 1.53 |
| AR + head scalars only | ar_head_heldout_scalar | 81.2 | 4.46 | 2.71 |

## Two-knob cost (×100), c_H=c_R=1 (`costs_cH=cR=1.<protocol>.cW=<c>`)
P1 = cross-fitted Platt on the test set (reproduces 14.06 vs 18.74, Δ −4.69). P2 = Platt fit on C only, prior shifted
to the test present rate: P2a known rate (0.549), P2b EM estimate (0.539 Glance / 0.545 AR) (`priors`).

| c_W | P1 AR+head | P1 Δ (Glance − AR+head) | P2a AR+head | P2a Δ | P2b AR+head | P2b Δ |
|---|---|---|---|---|---|---|
| 0 | 6.95 | +0.11 [−0.16, +0.35] | 7.27 | +0.25 [−0.00, +0.51] | 7.17 | +0.30 [+0.03, +0.56] |
| 0.5 | 8.21 | −0.07 [−0.37, +0.19] | 8.27 | +0.08 [−0.21, +0.35] | 8.22 | +0.02 [−0.28, +0.28] |
| 1 | 9.44 | −0.28 [−0.63, +0.08] | 9.40 | −0.21 [−0.59, +0.15] | 9.37 | −0.22 [−0.61, +0.13] |
| 2 | 11.56 | −0.81 [−1.32, −0.28] | 11.90 | −1.07 [−1.56, −0.59] | 11.86 | −1.07 [−1.55, −0.60] |
| 5 | 15.59 | −1.53 [−2.45, −0.64] | 16.36 | −1.89 [−2.71, −1.09] | 16.41 | −1.97 [−2.78, −1.16] |

- c_W=5, P1: the head lowers AR cost by 3.16 [2.39, 3.92] (`delta_ar+heldout_full_minus_ar`); Glance stays 9.8% cheaper (was 25%).
- Glance two-knob under P2a: 7.52 / 8.35 / 9.19 / 10.82 / 14.46; 11.6% (P2a) / 12.0% (P2b) cheaper than AR+head at c_W=5.
- P2 weak spot: C's box-error rate ≈ 2× the test's (10.4% vs 5.6%), so κ calibrated on C is pessimistic (head ECE 3.05 under P2 vs 0.50 under P1).
- Part of the remaining gap is localisation: AR base box error 5.6% vs Glance 4.9%.
- Head validation AUROCs 89.7 / 80.9 / 89.4: `head_scores_info.json <variant>.best_val_auroc`.
