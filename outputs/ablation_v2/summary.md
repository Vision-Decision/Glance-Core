# ablation_v2 summary

Two Colab G4 VMs (abl2-a on acct-0, abl2-lat on acct-8), both stopped at the end. abl2-b (from the previous agent) no
longer existed and could not be recreated (3-VM per-account cap). Rows were analysed on CPU (numpy) only.

## TL;DR
1. Exact MAP and corrected beam search confirm format-dependent refusal: digits ~34-41%, hier ~23-25%,
   atomic-1000 ~12%, atomic-100 ~4-5% of present targets; length-normalised beam removes it (1-3%). The 4.7% beam-4
   pilot was a pruning artifact. Refinement of one fixed model raises exact-MAP refusal 4.6 -> 10.7 -> 18.1 -> 29.4% (k=1/2/4/10).
2. Prop. 1 two-knob rule: vs gate-only it lowers cost only when c_W >= c_R (up to -24% at c_W=5); with both sides
   Platt-calibrated it beats the AR-digits analogue by -3.8% / -10.5% / -25% at c_W = 1 / 2 / 5 (CIs exclude 0),
   is tied at c_W <= 0.5, and at 5% training negatives loses for c_W <= 1.
3. Fair latency: Glance 67 ms flat vs vLLM+prefix-cache 174 / 218 / 331 ms at N = 1 / 8 / 32 (2.6 / 3.3 / 4.7x); batched
   8x8: 1.9x throughput; 8 images x 1 question only 1.2x. An AR presence-only read-out costs the same as Glance (67 ms).
   Mixed box/yes-no/choice/score: 2.5-3.2x.
4. Agent-style RCPS: on RefCOCO/gRefCOCO Glance certifies the most coverage (53.5% at <=5% wrong answers vs 51.6% AR
   margin); on GUI Glance v0 is too inaccurate (ScreenSpot-Pro ~4% coverage at 5%), and on VenusBench the generative base
   with a P(not none) score certifies more (18.4% vs 5.8%).

## 4. Decisive: exact MAP and corrected beam search, all 4 AR formats (audit Q1/Q2, round 3 item 1)

Files: `beam_fix/beamfix_summary.json`, `beam_fix/beamfix_refusal.png`, raw rows `raw/exactmap/`, `raw/beam_fix/`.
Code: `ablation/exactmap_ab.py` (exact best-first search pruned by log q(none <end>); accepts as soon as one complete
box beats "none", refuses when every prefix above that bound is exhausted; 10,000-expansion budget per question,
"undecided" otherwise), `ablation/incdec.py` + `ablation/beam_ab.py --keep_none` (HF beam semantics but "none <end>"
always a finished hypothesis; B = 4/8/16; lp0 = raw log-prob, lp1 = divided by length). Eval: RefCOCO val present
(single-target) and gRefCOCO val no-target, image subsamples (beam: stride 4 digits/hier, stride 2 atomics;
exact MAP: stride 16 digits/hier, stride 8 atomics). Present-target refusal %, [95% CI]:

| head | P(none)>=.5 | greedy | HF beam-4 (none pruned) | **beam-4 / 8 / 16, raw** | beam-16 len-norm | greedy-cand. MAP proxy | **exact MAP** | N-acc exact MAP |
|---|---|---|---|---|---|---|---|---|
| AR-digits | 0.4 | 0.6 | 22.1 | **39.2 / 38.9 / 38.2** [36.1, 40.1] | 1.0 | 40.8 | **[34.0, 41.3]** (34.0 proven + 16.5 undecided; capped by proxy) | 96.4 |
| AR-hier | 0.4 | 0.7 | 13.8 | **22.6 / 22.4 / 22.3** [20.7, 23.9] | 2.1 | 22.8 | **[23.1, 24.8]** | 95.1 |
| AR-atomic-1000 | 0.6 | 1.9 | 4.2 | **10.4 / 10.4 / 10.3** | 3.1 | 10.5 | **12.3** (+0.9 undecided) vs proxy 12.5 on the same subset | 91.9 |
| AR-atomic-100 | 0.4 | 0.8 | 3.6 | **3.9 / 3.9 / 3.9** [3.4, 4.4] | 1.2 | 3.9 | **4.6** (= proxy 4.6 on the same subset, 0 undecided) | 85.4 |

(Exact-MAP rows are on their own subset; proxy on that subset: digits 41.3, hier 24.8, atomic-1000 12.5, atomic-100 4.6.)

- **Format dependence survives exact MAP and real beam search**: digits ~34-41% > hier ~23-25% > atomic-1000 ~12% >
  atomic-100 ~4-5% of present targets refused, same ranking and magnitudes as the greedy-candidate proxy. The proxy is
  tight: exact MAP confirms 236/262 (digits), 161/164 (hier) and 63/63 (atomic-100) decided proxy refusals.
- The pilot's 4.7% beam-4 refusal is a pruning artifact: "none" falls outside HF's top-2B first tokens for 72% (digits),
  82% (hier), 89% (atomic-100) of present targets at B=4. Keeping "none" as a finished hypothesis restores MAP behaviour
  and beam converges to the proxy for B = 4..16.
- **Length normalisation (lp1, HF default) erases the effect** (1-3% refusal, N-acc 79-87%): the score definition of the
  decoder, not only the format, sets the refusal rate. P(none) thresholding and greedy are format-independent (<2%).
- MAP-refused present targets are mostly well localized by the greedy box (digits Acc@0.5 88%, hier 90%, atomic-100 79%).

### 4c. Prop. 2 fixed-model refinement test (original item 4), exact MAP — `raw/exactmap/ar_atomic100_neg25_split*/`

Same trained AR-atomic-100 model, every coordinate token split into k equal sub-forms (q/k each; no retraining);
exact MAP over the refined space (`exactmap_ab.py --split_k k`), stride 8 (1,362 present / 1,067 no-target):

| k | 1 | 2 | 4 | 10 |
|---|---|---|---|---|
| exact-MAP refusal of present targets | 4.6% | 10.7% | 18.1% | 29.4% |
| N-acc | 85.4% | 92.3% | 94.6% | 96.6% |
| P(none) >= .5 refusal (invariant by construction) | 0.7% | 0.7% | 0.7% | 0.7% |

Refinement alone, at fixed model probabilities, raises exact-MAP refusal 6x - Prop. 2 holds exactly. (The earlier
HF-beam split runs in `beam/beam_summary.json` stay flat at 3.3% because HF beam prunes "none"; ignore them.)

## 2. Prop. 1 two-knob rule (round 3, item 2) — `twoknob/twoknob.json`, `twoknob/twoknob.png`, code `ablation/twoknob_v2.py`

gRefCOCO-val-merged (10,834 present + 8,905 absent). Expected cost per item (x100), c_H = 1; rules tuned by 2-fold
image-level cross-fitting (out-of-sample); 95% CI = image bootstrap.

| c_R, c_W | Glance gate only | Glance g*kappa (Chow) | **Glance two-knob** (Prop. 1 family) | two-knob minus gate [95% CI] | AR-digits margin sweep | AR-digits 1-P(none) | AR-digits greedy-cand. MAP |
|---|---|---|---|---|---|---|---|
| 1, 0   | 7.13 | 9.49 | 7.06 | -0.07 [-0.20, 0.05] | 6.84 | 6.97 | 22.74 |
| 1, 0.5 | 8.26 | 10.10 | 8.20 | -0.06 [-0.22, 0.10] | 8.11 | 8.32 | 23.00 |
| 1, 1   | 9.51 | 10.62 | 9.21 | **-0.31 [-0.53, -0.08]** | 9.39 | 9.62 | 23.26 |
| 1, 2   | 11.61 | 11.85 | 10.73 | **-0.88 [-1.17, -0.58]** | 12.01 | 12.39 | 23.79 |
| 0.5, 2 | 9.14 | – | 8.00 | **-1.14 [-1.49, -0.79]** | 9.63 | – | – |

- The Prop. 1 rule helps exactly when box errors cost something (c_W >= c_R): -0.3 to -1.1 cost points vs gate-only,
  CIs exclude 0; it is never worse. With c_W <= 0.5 it reduces to the gate (no gain).
- The pure Chow rule on g*kappa is the worst Glance rule at every cost (as Prop. 1 predicts: not Bayes).
- vs AR-digits margin sweep: Glance two-knob is better for c_W >= 1 (10.73 vs 12.01 at c_W=2) and when refusals are
  cheap (c_R=0.5); at c_R=1 with c_W <= 0.5 the AR margin is equal or slightly better (6.84 vs 7.06 at c_W=0; CIs overlap).
- Plug-in (untuned) Prop. 1 with raw heads costs ~11-14 (vs ~7-11 tuned): the heads are not calibrated to this 45%-absent
  mix (trained at 25%), so thresholds must be calibrated; the direction (two knobs) is what matters.
- Risk-coverage along the family w*g + (1-w)*g*kappa: AURC 16.21 (w=0, g*kappa) / 16.20 (w=1, g) but **15.75 at
  w=0.95** — a mixture beats both endpoints; the "identical 16.21" is two different curves crossing (see item 5).
- 5% negatives: tuned two-knob = gate except c_W=2 (-0.28..-0.37).

## 2b. Audit Q7: fair two-knob comparison (both sides Platt-calibrated, Prop. 1 plug-in, no threshold tuning)

Cross-fitted (2-fold, image halves) Platt scaling: Glance eta <- logit(g), kappa <- logit(kappa_hat) (fit on present
items, target IoU>=.5); AR eta <- margin log max_c q(c) - log q(none), AR kappa <- log length-normalised box
confidence. Then answer iff (c_R - c_W + c_H) eta + c_W eta kappa > c_H. Cost x100 on gRefCOCO-val-merged (45% absent);
Delta = Glance - AR, paired image bootstrap.

| c_R, c_W | Glance 2-knob | Glance eta only | AR-digits 2-knob | AR eta only | Delta [95% CI] | rel. |
|---|---|---|---|---|---|---|
| 1, 0 | 7.06 | 7.06 | 6.95 | 6.95 | +0.11 [-0.16, 0.35] | +1.5% |
| 1, 0.5 | 8.13 | 8.19 | 8.24 | 8.21 | -0.10 [-0.40, 0.17] | -1.3% |
| 1, 1 | 9.16 | 9.32 | 9.52 | 9.46 | -0.36 [-0.72, -0.01] | -3.8% |
| 1, 2 | 10.75 | 11.59 | 12.00 | 11.98 | -1.26 [-1.79, -0.72] | -10.5% |
| 1, 5 | 14.06 | 18.38 | 18.74 | 19.51 | -4.69 [-5.87, -3.60] | -25.0% |
| 0.5, 5 | 9.84 | 16.02 | 14.55 | 16.52 | -4.71 [-5.70, -3.85] | -32.4% |

- The AR "two-knob" (calibrated margin x calibrated box confidence) barely improves on AR eta-only (18.74 vs 19.51 at
  c_W=5): the AR box confidence is a weak kappa (positive-only AUROC 66-75 vs 91.6). Glance's kappa cuts its own cost by
  24% at c_W=5. Hier / atomic-100 / atomic-1000 Platt 2-knob are all worse than digits (e.g. 19.6 / 19.7 / 19.7 at c_R=1,c_W=5).
- With c_W = 0 the AR head is as good or slightly better (CI includes 0); at 5% training negatives Glance is worse
  for c_W <= 1 (+0.5..+0.8, CIs exclude 0) and better only for c_W >= 2 (-3.3 at c_W=5).

## 1. Fair latency (RTX PRO 6000 96GB, G4; 100 RefCOCO-val images, <=640 visual tokens, LoRA merged)

See `latency/latency_table.md`, `latency/latency_vs_N.png`, raw `latency/*.json`.

Per-call median ms (one image, N questions):

| system | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | 8 img x N=1 | 8 img x N=8 |
|---|---|---|---|---|---|---|---|---|
| Glance (ours, eager PyTorch) | 67 | 67 | 67 | 67 | 67 | 70 | 311 (26 q/s) | 346 (186 q/s) |
| AR presence-only q(t1=none), one padded block pass (same engine) | 67 | 67 | 67 | 67 | 68 | 71 | 314 (26 q/s) | 350 (183 q/s) |
| AR presence-only, blocks started per length group (as the incremental decoder needs) | 75 | 102 | 129 | 156 | 160 | 163 | 408 | 545 |
| AR-digits, incremental KV/GDN-state cache (same engine) | 446 | 462 | 517 | 550 | 569 | 603 | 828 | 1405 (44 q/s) |
| Qwen3.5-4B generative, vLLM 0.34 + automatic prefix caching (CUDA graphs) | 174 | 186 | 200 | 218 | 255 | 331 | 375 (21 q/s) | 666 (95 q/s) |
| same, staged (first question alone, rest after its first token) | 174 | 197 | 213 | 235 | 276 | 349 | 375 | 651 |
| vLLM without prefix caching | 171 | 196 | 236 | 309 | 461 | 755 | 373 | 1419 (45 q/s) |

Ratio vLLM+PC / Glance: 2.6x (N=1), 3.3x (N=8), 4.7x (N=32); batched 8 img x 8 q: 1.9x (throughput 186 vs 95 q/s);
8 img x 1 q: only 1.2x. Time-to-first-answer: Glance 67-70 ms vs vLLM 173 (N=1) / 192 (N=8) / 289 (N=32).
vLLM prefix cache is active: cached prompt tokens 282/423/495/547 of 691 for N=2/4/8/32 (the image+template prefix
~565 tokens is reused by every question after the first). Generated answers ~19.4 tokens.
Caveats: Glance and the in-engine AR heads run in eager PyTorch (no CUDA graphs); vLLM's per-token cost (~7 ms) is
~3x lower than our eager incremental step (~20 ms), so "AR-digits incremental (our engine)" overstates AR cost; the vLLM
row is the optimized-AR reference. vLLM timing includes its HF image processor; ours excludes preprocessing.

Reading: (i) against prefix-cached, batched vLLM decoding Glance is 2.6x (N=1) to 4.7x (N=32) faster per image and has
2.6-4.1x lower time-to-first-answer; at 8 images x 8 questions throughput is 1.9x. With 8 images x 1 question the gap is
only 1.2x (our eager engine is slow at multi-image batches: 39 ms/image). (ii) An AR head's presence-only read-out
q(t1=none) costs exactly as much as Glance (67 ms) - it gives the refusal decision but no box; any box costs ~19 decode
steps. So the speed advantage is "zero decode steps", not the readout per se; flatness in N comes from the shared prefix
and applies to AR too (vLLM+PC grows 174->331 ms from N=1 to 32, without PC 171->755 ms).

## 1b. Mixed typed prefill latency (round 3, item 4) — `latency/mixed_ours.json`, `latency/mixed_vllm_pc.json`

Code `ablation/latency_mixed.py` (Glance v1 checkpoint, dv/ model, LoRA merged, eager), `ablation/latency_vllm.py --mixed`,
workload `ablation/mixed_q.py`: one RefCOCO-val image (<=640 visual tokens), Q questions cycling box / yes-no / choice
(3 options) / score (5 levels), same prompt bodies for both systems; 100 images (50 for Q>=16), batch 1.

| Q | Glance v1, one prefill (median / p90 ms) | vLLM + prefix cache, generative (median / p90) | vLLM time-to-first-answer | ratio |
|---|---|---|---|---|
| 1 | 67.4 / 68.3 | 173.0 / 184.2 | 173.0 | 2.6x |
| 2 | 67.1 / 68.7 | 171.2 / 180.3 | 53.6 | 2.6x |
| 4 | 67.2 / 70.5 | 215.3 / 218.1 | 57.4 | 3.2x |
| 8 | 72.0 / 76.8 | 231.1 / 234.7 | 66.9 | 3.2x |
| 16 | 92.5 / 94.7 | 258.1 / 266.9 | 86.7 | 2.8x |
| 32 | 128.4 / 141.1 | 322.7 / 335.7 | 126.9 | 2.5x |

Glance answers all Q typed + box questions in one call (TTFA = call time); its cost grows from Q=8 on because typed
blocks are long (option lists). vLLM generates ~11.4 tokens per mixed question on average (box ~19, typed ~2-4), so its
first (typed) answer arrives early (54-127 ms), but the last answer takes 2.5-3.2x longer than Glance's whole batch.

## 3. Agent-style risk-controlled answering (round 3, item 3) — `agent/agent_rcps.json`, code `ablation/agent_rcps_v2.py`

RCPS (Clopper-Pearson UCB, delta=0.1) on a calibration half, 200 random 50/50 splits; loss = answered and wrong
(absent, IoU<0.5, or click outside GT). Coverage = answered share on the test half (test risk always <= alpha on average).

| set | system | cov @2% | cov @5% | cov @10% | latency/query |
|---|---|---|---|---|---|
| gRefCOCO-val-merged | Glance two-knob (w=.9) | 40.9 | 53.5 | 61.2 | 67 ms/call for 8 q (bs1) |
| | Glance gate | 38.7 | 52.5 | 61.1 | |
| | AR-digits margin | 37.6 | 51.6 | 60.7 | 550 ms (our engine) |
| | AR-digits greedy / greedy-cand. MAP (fixed points) | 0 / 33.8 | 0 / 33.8 | 0 / 33.8 | |
| ScreenSpot-v2 | Glance v0 @640 tok (best score) | 7.3 | 13.3 | 23.4 | ~67 ms |
| | Qwen3.5-4B generative @4000 tok (no score) | 0 | 0 | 98.5 | 543 ms median |
| ScreenSpot-Pro | Glance v0 @640/@4000 | 1.4 | 4.0-4.3 | 8.6-8.9 | |
| | generative base (no score) | 0 | 0 | 0 | 979 ms |
| VenusBench-GD (2,116 common rows) | Glance v0 @4000 (two-knob) | 2.0 | 5.8 | 11.7 | |
| | generative base, P(not none) score | 7.8 | 18.4 | 31.9 | 917 ms |
| | generative base, fixed | 0 | 0 | 0 | |

Reading: on RefCOCO/gRefCOCO a scored rule can be certified at 2-10% wrong-answer rate, the two-knob score gives the
largest certified coverage (+2-3 pts over the AR margin) at ~8x lower latency. A generative grounder with no score can
only be certified when its overall error is below alpha (ScreenSpot-v2 at 10%), otherwise RCPS must abstain on all.
But on GUI, Glance v0 is simply too inaccurate (click acc. 29% SS-v2, ~1-2% SS-Pro): certified coverage is tiny, and the
generative base with a token-probability score beats it on VenusBench (18.4% vs 5.8% at 5%). No GUI AR head and no v1/v1.1
GUI rows exist locally, so the agent number that favours Glance is RefCOCO-only.

## 5. Selective-prediction reanalysis (CPU) — `selective/selective.json`, `selective/selective_extra.json`, `selective/*.png`

Code: `ablation/selective_v2.py` (previous agent), `ablation/selective_extra_v2.py`.
- Risk-coverage incl. box errors: AURC gate 16.20 [15.66, 16.70], gate*kappa 16.21; AR-digits 1-P(none) 16.64,
  P(s_hat) 24.55, (1-P(none))*conf 20.94; hier 16.57; atomic-100 16.52; atomic-1000 17.13.
- **AURC(g) = AURC(g*kappa) is not a bug**: exact values 0.162041 vs 0.162074; the orderings differ substantially
  (Spearman 0.90, 12% of item pairs ordered differently; risk differs by up to 6.4 pts locally); g*kappa is better at low
  coverage (30%: 3.45 vs 3.53%) and worse at 40-60%, the areas happen to cancel. With presence-only error the two AURCs
  differ clearly (13.73 vs 14.53); at 5% negatives 16.53 vs 17.17. Mixture w=0.95 gives 15.75 (item 2).
- Positive-only (present targets, error = IoU<0.5): AURC gate 1.81, gate*kappa 0.69, kappa alone 0.66 (AUROC 91.6);
  AR-digits best 2.22 (1-P(none)), P(s_hat) 2.89 (AUROC 70.9); hier 2.16; atomic-100 2.14; atomic-1000 2.53.
- AR margin sweep (logit bias b on "none" at step 1): at matched N-acc the gate has lower selective risk at every level
  (N-acc 90%: risk 11.7 vs 12.3 digits / 12.3 hier / 12.6 atomic-100 / 12.6 atomic-1000; N-acc 95%: 7.5 vs 8.0-8.7)
  with similar present refusal (4.8 vs 4.4%). AURC of the margin score: see `margin_sweep.*.aurc_margin`.
- Greedy-candidate-MAP-refused present targets (seed 0): digits refuses 39.9% of present; refused mean IoU 0.81 /
  Acc@0.5 88.3% vs accepted 0.92 / 98.4%; hier 22.0% refused (0.79 / 86.4%); atomic-1000 9.9% (0.72 / 78.1%);
  atomic-100 3.4% (0.69 / 75.9%); digits 5% neg 15.2% (0.72 / 79.4%). Most MAP-refused targets are localized correctly.

## 6. Multi-seed (original item 2; partially done) — `seeds/seeds.json`, `seeds/seeds.png`, code `ablation/seeds_v2.py`

Done: seed 1 for Glance and AR-digits at 25% negatives (seed 0 = original). **Not done** (budget/VM cap; dropped when
the round-3/audit items took priority): seed 2 at 25%, seed 1 at 5% (ar_digits_neg25_s2 was started then stopped;
abl2-b no longer existed when I took over and the 3-VM per-account cap blocked a replacement).
Mean ± std over seeds {0,1} [95% CI of the seed mean, image-cluster bootstrap]:

| metric (%) | Glance | AR-digits | AR - Glance [95% CI] |
|---|---|---|---|
| RefCOCO val Acc@0.5 | 95.05 ± 0.06 | 94.41 ± 0.05 | -0.64 [-1.03, -0.27] |
| mean IoU | 84.05 ± 0.03 | 87.77 ± 0.04 | |
| N-acc (gate / greedy) | 76.02 ± 0.48 | 77.56 ± 0.63 | +1.54 [1.17, 1.89] |
| N-acc (AR P(none)>=.5) | | 75.55 ± 0.40 | -0.47 [-0.83, -0.12] |
| N-acc @ 1% present refusal | 82.01 ± 0.95 | 81.95 ± 0.25 | -0.06 [-1.03, 0.86] |
| presence AUROC | 97.67 ± 0.19 | 97.64 ± 0.11 | -0.03 [-0.13, 0.06] |
| present refusal, greedy-cand. MAP | | 37.64 ± 3.25 (39.94 / 35.34) | +37.2 [35.8, 38.7] |

The seed-0 conclusions hold at seed 1: matched-threshold presence quality is indistinguishable, Glance has +0.6 Acc@0.5,
AR has the tighter boxes, and MAP refusal for digits is 35-40% in both seeds.

## 7. Not done / caveats
- Multi-seed: seed 2 at 25% and seed 1 at 5% not trained (budget, VM cap); only seeds {0,1} at 25%.
- Item 6 (AR-digits + typed heads on the v1 mixture): not started (budget).
- Audit Q4 (generative Qwen3.5-4B on the remaining ~4,049 VenusBench items): not done here; no GPU time left in budget.
- Exact MAP runs on image subsamples (stride 8/16) with a 10k-expansion budget; digits/hier have 16-21% undecided present
  questions, bracketed by the proxy (an upper bound).
- Latency: Glance and in-engine AR heads run eager PyTorch (no CUDA graphs); vLLM uses CUDA graphs and includes its image
  preprocessing; our timings exclude preprocessing.
- `selective_v2.py` and the stride-3 HF beam run (`beam/`) were written by the previous agent; `beam/` split rows are
  invalid as a refinement test (pruned "none").
- Large row files under `raw/` are gzipped (`gunzip` before re-running `beamfix_analyze.py` / `beam_v2_analyze.py`).
