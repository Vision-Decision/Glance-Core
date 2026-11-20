# More than 26 options: design proposal (no training code changed)

## Problem

The v1 typed readout (`dv/model/typed.py`) is a softmax over the LM-head rows of `A..Z` plus `No`/`Yes` (N_LAB = 28),
with a zero-init residual, read at the block's last token. A `choice` therefore covers at most 26 options in one read.
The board allows 2 to 255 options per question.

What really exceeds 26 options on the public suite (measured, see `GAPS.md` and `suite_stats.json`): BANKING77 (77),
CLINC150+OOS (151), ChessBench (all legal moves), POP909-CL (the chord vocabulary) and API-Bank (the catalog). Those
are the large-option benchmarks.

**Correction to the brief: ToolRet and BRIGHT are not 32-way choices.** Each candidate document is its own binary
question (`criteria = {yes, no}`), packed into chunks of at most 60k cl100k tokens per request, and nDCG@10 is computed
over P(yes). The same holds for BFCL tool selection and SATA-Bench "select all". In those benchmarks the problem is
long and numerous blocks, not the option count (see `GAPS.md`).

## Evidence from the board (public 0.3 native metric per benchmark, not chance-corrected; 2026-11-12 snapshot)

| entrant (4B unless noted) | >26-option mechanism | BANKING77 | CLINC150 | ChessBench | POP909 | API-Bank |
|---|---|---:|---:|---:|---:|---:|
| KnowLine 4B Gen5.5 (Full 55.99) | label-token readout (card does not say how >26 is labelled) | .903 | .926 | .228 | .843 | .890 |
| StartLux v1.1 4B (Full 54.80) | **chunked group rounds**: groups of at most 25, top 3 of each to a final read | .907 | .935 | .221 | .803 | .870 |
| pplx Decider Flash v1.3 (Full 55.14) | **255-way readout** at a marker token (`[255, 2560]` head) | .912 | .935 | .225 | .271 | .845 |
| chinese-decision-pointer-4b (Full 53.96) | **pointer head**: marker after each option, readout-marker x option-marker score | .888 | .948 | .122 | .581 | .845 |
| GLiDE 27B (Full 73.37) | 255-way decision readout | .908 | .963 | .248 | .821 | .856 |

Readouts don't separate these models much on intent classification: they all land at 0.89 to 0.95. Two other things
stand out:
- Chunked letter reads (StartLux) are no worse than single-read designs anywhere.
- The 255-way and pointer heads drop sharply on POP909 (0.27 and 0.58, against 0.80 to 0.84 for letter reads). The
  pointer head also drops on ChessBench (0.12 against 0.22).

Training data differs across these entrants, so this is suggestive, not causal. It does show that option (c) is not a
handicap in practice.

## Options

### (a) 255-way label readout

Extend the label set from 26 letters to 255 codes and read them at one token (pplx Decider style). This needs 229 more
single-token codes. Qwen tokenizes digits one per token, so the codes must be multi-letter vocabulary pieces or
reserved special tokens.
- **Accuracy**: no pretrained prior for codes past `Z`. The TypedHead's "LM rows of the label tokens" init only holds
  for `A..Z`, so codes 27 to 255 start from embeddings that mean nothing as answer labels. Binding code to option index
  across 255 lines is a pure attention-lookup skill the 4B must learn from many-option data.
- **Calibration**: fine once trained, with one temperature.
- **Latency**: one read, the cheapest option.
- **Fit**: drop-in for the shared-prefix forward (bigger `E` buffer and `res` layer, `lab_idx` up to 255). It breaks
  the "readout = base LM at init" property for codes past `Z`.
- **Effort**: about 2 days of code (tokens, head, collate, prep for many-option records) plus about 1 day of data, then
  a full retrain (about 85 CU at the v1 recipe). Highest risk of the three.

### (b) Set-native option scoring (pooled span or marker pointer)

Score every option from its own representation: pool its span, or place a marker after it. Read the option against
the question readout, `s_i = <W_q h_read, W_k h_opt_i> / sqrt(d) + b`, then softmax over N. This is the
chinese-decision-pointer design, and the kind of set-native head the brief attributes to Jeva (arXiv:2610.18773, which
I did not read).
- **Accuracy**: natural for any N. Options can see each other in the full-attention layers thanks to our block-bidir
  mask, and order bias is low. But there is no pretrained prior: the head starts from scratch, and on the board this
  family is weaker on POP909 and ChessBench.
- **Calibration**: good with CE plus Brier and one temperature.
- **Latency**: one read.
- **Fit**: needs the per-token block hidden states, which `SharedPrefixLM` already returns (`hq [M,Tq,D]`), plus span
  or marker indices from collate. GDN layers stay causal, so early options never see later ones there.
- **Effort**: about 3 to 4 days of code (span or marker bookkeeping in `mixed.py`, a head, losses, distillation from
  the letter head for N <= 26 so behaviour is kept), many-option training data, and a full retrain. Medium-high risk.

### (c) Chunked or hierarchical scoring on the shared prefix (recommended)

Split the options, in their original order, into ceil(N/26) near-equal chunks. Each chunk is one more question block
on the same prefix, so round 1 costs nothing extra: it runs in the same forward as the request's other questions. The
top-k of every chunk (k = min(3, 26 // chunks); 77 options give 3 chunks with k = 3 and 9 finalists, 151 options give
6 chunks with k = 3 and 18 finalists, 255 options give 10 chunks with k = 2 and 20 finalists) go to a final read.

Calibration across chunks, as now in `di/engine.py`:
- M_c is the final-round mass of chunk c's finalists.
- A non-finalist o of chunk c gets M_c · P_c(o).
- The finalists share the remaining `1 - r` in final-round proportions, where r = Σ_c M_c · (1 − P_c(finalists of c)).
- The probabilities sum to 1, every option gets a probability, and the final head-to-head read decides the argmax.

`chunk_mode=concat` (one read, global softmax over the concatenated chunk logits) is kept as a cheaper ablation.

- **Accuracy**: every decision is a 26-way-or-smaller letter read, exactly what v1 was trained on. The final read
  compares the strongest candidates directly, and the board shows the same scheme (StartLux) on par with the best.
  Losses come from an early-round miss: the gold falls out of its chunk's top-k. That is measurable, and gets smaller
  with k and with training on chunked lists.
- **Calibration**: two temperatures (chunk read, final read) can be fitted on a held-out many-option split. The
  composition is coherent: the chunk-level mass comes from the final read. The scored DI metrics on these benchmarks
  are accuracy and macro-F1, which only see the argmax.
- **Latency**: +1 forward only for >26-option questions (about +70 to 90 ms on the RTX PRO 6000, more with a long
  prefix since we recompute it). On v1's ~70 to 90 ms/request base this is far below the gate (see the
  `GAPS.md` latency table). Caching the prefix's GDN state and KV would make round 2 cost only its block; that is a
  backbone change, not done here.
- **Fit**: uses the v1 heads unchanged, keeps the LM-row prior, and needs no training. "No option filtering" is met:
  every option is shown to the model in round 1 and every option is priced. The mechanism is declared in
  `provenance.large_option_fallback`.
- **Effort**: **done now** as the declared fallback (`di/engine.py`, about 80 lines). Hardening:
  1. fit chunk and final temperatures on a held-out many-option split from non-DI sources (0.5 day);
  2. measure the round-1 recall@k of gold on a held-out many-class set and tune k (0.5 day);
  3. in v1.x data prep, add many-option records rendered as chunk reads and as final reads, so both stages are
     in-distribution (1 day prep, then the usual retrain);
  4. optionally, prefix-state caching for round 2 (1 to 2 days, `dv/model/backbone.py`).

## Recommendation

**Use (c) now and keep it.** It is the only option with zero training, it uses the head we already calibrated, it
costs one extra short forward only on the 10.3% of requests that need it, and the board shows it is not the
bottleneck. Spend the next training budget on many-option and chunk-format data (step 3) plus the general DI-shaped
data that actually limits a 4B (knowledge, long-context NLI, multi-question tool rows; see `GAPS.md`), not on a new
head.

Revisit (b) only if round-1 recall@k turns out to be the dominant error on the private same-skill tests (we can't see
those, so the proxy is a held-out many-class set). Then build it as an extra head trained by distillation from the
letter head, so ≤26-option behaviour is preserved. Option (a) is not recommended: highest risk, no prior past `Z`,
and the board's 255-way entrant shows the largest drops on the large-vocabulary tasks.
