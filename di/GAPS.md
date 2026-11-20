# Gap analysis: Glance v1 against the Decision Index public suites

Measured 2026-11-18 on suites rebuilt locally from the pinned sources with the kit @25c6ec7:
- **Text 0.3**: added rows and GSM8K rows are byte-identical to the lab's. The base rows differ only in the SGD normalized file (see `RUN.md`, "Suite integrity"). The text statistics below count 140,620 rows, which includes the 442 excluded questions; 140,178 are scoreable.
- **Vision**: 11,410 of 12,210 rows. Winoground (800 rows) is gated, and this HF account has not accepted its terms. InfographicVQA and CORD images differ by resample bytes under Pillow 13 (the kit pins 9.5 or 12.3); their shapes are unaffected.

Tokens are counted with **our tokenizer and our exact rendering** (`di/engine.py`, via `di/suite_stats.py`). Raw numbers are in `di/suite_stats.json`.

## Headline

| | text (37 public benchmarks + 6 shown) | vision (11) |
|---|---|---|
| Primitives on the board | only `choice` (2 to 255 options) and `noul`. No `score`, no `box` | `choice` only, letter keys, 1 to 2 images per row in the built sets (kit allows up to 4) |
| Answerable by v1 as-is (≤26 options, within limits) | **126,120 / 140,620 rows (89.7%)**, 32 of 37 index benchmarks | **100%** (max 12 options, MMMU-Pro) |
| Needs >26 options | **14,500 rows (10.3%)**: CLINC150 5,500 (151), BANKING77 3,080 (77), ChessBench 3,412 of 5,000 (≤81), POP909 2,000 (129), API-Bank 508 (53). All are answered now through the chunk tournament fallback (`LARGE_OPTIONS.md`) | 0 |
| Over our declared limits (prefix 65,536 / block 32,768 tokens) | **0** rows. Largest request: ToolRet, 71,678 tokens over 32 blocks (split into several forwards, nothing cut). Largest block: 30,679 tokens | 0. Largest prefix 3,207 tokens (2 × 1,600 image tokens) |
| Over v1's *training* caps (state 1,536 tokens) | 12,951 rows, mostly RouterBench 9,929 (not counted), POP909 1,622, API-Bank 508, When2Call 250, NLI4CT 172, BFCL 127, SATA 122, RAGTruth 111, ContractNLI 78. They run (no truncation), but out of training distribution | image tokens: v1 trained ≤640, the board runs at ≤1,600. **1,837 rows** have a source image larger than 1.6 MP (downscaled to the cap), and many more exceed v1's 640-token training size |
| Estimated latency, one RTX PRO 6000 (model below) | p50 **85 ms**, mean **128 ms**, p80 **164 ms**; 0.34% of rows over 1 s (ToolRet, BRIGHT, POP909 tail) | p50 98 ms, mean 102 ms, p80 120 ms |

"Answerable" means the harness returns a valid full distribution. It says nothing about accuracy.

### Correction to the brief

ToolRet and BRIGHT do **not** pose a 32-way choice. Each candidate is its own `yes`/`no` choice, up to 32 per request (chunked at about 60k cl100k tokens), and the score is nDCG@10 over P(yes). BFCL, SATA-Bench, ACOS and iSarcasm are posed the same way: several binary questions over one state. This is native to the shared-prefix forward. The cost is long or numerous blocks, not the option count.

## Text benchmarks

Latency model (an assumption, to be calibrated with `di/latency_probe.py` on the GPU):
- fixed 70 ms per request (v1 measured 67 ms for `text_1q`);
- + tokens / 20k tok/s, + 0.3 ms per block, + 0.03 ms per image token;
- \>26-option rows add a second forward (≈0.6 × fixed + prefix).

`(not counted)` marks a benchmark that is shown but is outside the public index.

| area | benchmark | requests | primitive as posed | max options | requests >26 opts | tokens/request p50 / p95 / max | prefix >1536 (v1 train cap) | answerable now | est. ms p50 / mean / max |
|---|---|---:|---|---:|---:|---|---:|---|---|
| arts | BPoMP | 5000 | choice 2-way | 2 | 0 | 171 / 185 / 211 | 0 | yes | 78 / 78 / 80 |
| arts | Habermas Machine | 1676 | choice ≤5 | 5 | 0 | 882 / 1211 / 1527 | 0 | yes | 114 / 115 / 146 |
| arts | Humicroedit | 2628 | choice 2-way | 2 | 0 | 76 / 94 / 114 | 0 | yes | 74 / 74 / 76 |
| arts | New Yorker | 528 | choice ≤5 | 5 | 0 | 195 / 237 / 277 | 0 | yes | 80 / 80 / 84 |
| arts | POP909-CL | 2000 | choice ≤129 | 129 | 2000 | 6563 / 7854 / 19631 | 1622 | fallback (c) on all | 544 / 550 / 1850 |
| arts | cfcolor | 5000 | choice 2-way | 2 | 0 | 524 / 536 / 546 | 0 | yes | 96 / 96 / 97 |
| knowledge | BBH | 5507 | choice ≤18 | 18 | 0 | 174 / 330 / 615 | 0 | yes | 79 / 79 / 101 |
| knowledge | CLadder | 5000 | choice 2-way | 2 | 0 | 153 / 260 / 349 | 0 | yes | 77 / 78 / 87 |
| knowledge | CRUXEval | 570 | choice ≤10 | 10 | 0 | 163 / 278 / 849 | 0 | yes | 78 / 79 / 112 |
| knowledge | ChessBench | 5000 | choice ≤81 | 81 | 3412 | 1052 / 1408 / 2427 | 0 | fallback (c) on 3412 | 172 / 152 / 241 |
| knowledge | GPQA Diamond | 198 | choice ≤4 | 4 | 0 | 214 / 456 / 2776 | 0 | yes | 81 / 82 / 209 |
| knowledge | GSM8K | 2638 | choice ≤10 | 10 | 0 | 208 / 273 / 371 | 0 | yes | 80 / 80 / 88 |
| knowledge | HLE | 513 | choice ≤26 | 26 | 0 | 251 / 1246 / 13553 | 0 | yes | 82 / 92 / 747 |
| knowledge | HellaSwag | 10042 | choice ≤4 | 4 | 0 | 259 / 312 / 412 | 0 | yes | 83 / 81 / 90 |
| knowledge | MMLU-Pro | 12032 | choice ≤10 | 10 | 0 | 214 / 483 / 1706 | 0 | yes | 81 / 82 / 155 |
| knowledge | MuSR | 756 | choice ≤5 | 5 | 0 | 1004 / 1354 / 1565 | 0 | yes | 120 / 119 / 148 |
| knowledge | SATA-Bench | 1650 | 9 binary choices/req (yes/no per candidate) | 2 | 0 | 935 / 2650 / 5940 | 122 | yes | 118 / 135 / 371 |
| knowledge | WinoGrande | 1267 | choice 2-way | 2 | 0 | 67 / 76 / 84 | 0 | yes | 73 / 73 / 74 |
| knowledge (not counted) | ARC-Challenge | 1172 | choice ≤5 | 5 | 0 | 93 / 140 / 226 | 0 | yes | 74 / 75 / 81 |
| knowledge (not counted) | ARC-Easy | 2376 | choice ≤5 | 5 | 0 | 82 / 127 / 225 | 0 | yes | 74 / 74 / 81 |
| knowledge (not counted) | MMLU | 14042 | choice ≤4 | 4 | 0 | 106 / 346 / 1028 | 0 | yes | 75 / 77 / 121 |
| knowledge (not counted) | SimpleBench | 10 | choice ≤6 | 6 | 0 | 187 / 272 / 293 | 0 | yes | 79 / 80 / 84 |
| language | ACOS | 1565 | 58 binary choices/req (yes/no per candidate) | 2 | 0 | 7122 / 7213 / 7259 | 0 | yes | 445 / 412 / 452 |
| language | ANLI | 3200 | choice ≤3 | 3 | 0 | 155 / 195 / 270 | 0 | yes | 78 / 78 / 83 |
| language | ContractNLI | 123 | multi-q choice (17.0/req) | 3 | 0 | 3542 / 6458 / 9664 | 78 | yes | 252 / 269 / 558 |
| language | FinEntity | 979 | multi-q choice (2.2/req) | 3 | 0 | 230 / 504 / 1630 | 0 | yes | 82 / 83 / 156 |
| language | NLI4CT | 5500 | choice 2-way | 2 | 0 | 508 / 1496 / 3070 | 172 | yes | 95 / 101 / 223 |
| language | RAGTruth | 2700 | noul | 2 | 0 | 800 / 1551 / 2452 | 111 | yes | 110 / 114 / 192 |
| language | VAST | 3006 | choice ≤3 | 3 | 0 | 173 / 284 / 341 | 0 | yes | 78 / 79 / 87 |
| language | iSarcasmEval | 4600 | 3 binary choices/req (yes/no per candidate) | 2 | 0 | 70 / 315 / 429 | 0 | yes | 73 / 77 / 93 |
| retrieval | Amazon ESCI | 5000 | choice ≤4 | 4 | 0 | 353 / 974 / 2348 | 4 | yes | 87 / 92 / 187 |
| retrieval | BANKING77 | 3080 | choice ≤77 | 77 | 3080 | 1041 / 1092 / 1242 | 0 | fallback (c) on all | 165 / 165 / 175 |
| retrieval | BRIGHT | 307 | 32 binary choices/req (yes/no per candidate) | 2 | 0 | 17966 / 45802 / 62329 | 6 | yes | 977 / 1173 / 3196 |
| retrieval | CLINC150+OOS | 5500 | choice ≤151 | 151 | 5500 | 1862 / 1910 / 2006 | 0 | fallback (c) on all | 207 / 207 / 214 |
| retrieval | HoVer | 4000 | choice 2-way | 2 | 0 | 449 / 707 / 1015 | 0 | yes | 92 / 93 / 121 |
| retrieval | PhishNChips | 2000 | multi-q: 3 choice + 6 noul | 2 | 0 | 670 / 801 / 1226 | 0 | yes | 106 / 106 / 134 |
| retrieval (not counted) | SGD/SGD-X | 2500 | choice ≤4 | 4 | 0 | 881 / 1345 / 2014 | 22 | yes | 114 / 115 / 171 |
| tools | API-Bank | 508 | choice ≤53 | 53 | 508 | 7414 / 7929 / 8366 | 508 | fallback (c) on all | 819 / 826 / 914 |
| tools | BFCL | 1694 | 3 binary choices/req (yes/no per candidate) | 2 | 0 | 779 / 2151 / 8687 | 127 | yes | 109 / 118 / 515 |
| tools | Home appliance simulator | 88 | multi-q choice (18.1/req) | 9 | 0 | 2657 / 2797 / 2828 | 0 | yes | 208 / 208 / 217 |
| tools | ToolRet | 1013 | 30 binary choices/req (yes/no per candidate) | 2 | 0 | 10123 / 63040 / 71678 | 0 | yes | 585 / 1183 / 3661 |
| tools | When2Call | 3652 | choice ≤4 | 4 | 0 | 793 / 1783 / 5249 | 250 | yes | 109 / 114 / 332 |
| tools (not counted) | RouterBench | 10000 | multi-q choice (2.0/req) | 11 | 0 | 2519 / 3097 / 5494 | 9929 | yes | 196 / 201 / 345 |

## Vision benchmarks

Images run at native resolution capped at 1.6 MP (≤1,600 tokens of 32×32 px), never upscaled.

| benchmark | rows | images/row max | max options | image tokens p50 / max (1.6 MP cap) | rows with a source image >1.6 MP | tokens/request p50 / max | answerable now | est. ms p50 / mean |
|---|---:|---:|---:|---|---:|---|---|---|
| BLINK | 961 | 2 | 4 | 600 / 3200 | 51 | 723 / 3414 | yes | 94 / 117 |
| CV-Bench | 2038 | 1 | 6 | 260 / 1598 | 138 | 342 / 1668 | yes | 82 / 89 |
| CharXiv | 532 | 1 | 4 | 672 / 1024 | 0 | 759 / 1136 | yes | 94 / 94 |
| InfographicVQA | 244 | 1 | 4 | 1530 / 1600 | 144 | 1602 / 1700 | yes | 119 / 113 |
| KIE (CORD+FUNSD) | 2465 | 1 | 4 | 775 / 1600 | 342 | 853 / 1694 | yes | 97 / 102 |
| MMMU-Pro vision | 1730 | 1 | 12 | 1536 / 1600 | 1015 | 1612 / 1682 | yes | 120 / 113 |
| Mind2Web | 973 | 1 | 4 | 1360 / 1593 | 6 | 1498 / 1697 | yes | 118 / 118 |
| Moderation (Hateful Memes) | 1213 | 1 | 2 | 400 / 625 | 0 | 445 / 670 | yes | 84 / 82 |
| R-Bench-M | 665 | 1 | 6 | 120 / 1600 | 20 | 426 / 2316 | yes | 89 / 92 |
| RealWorldQA | 589 | 1 | 4 | 1305 / 1598 | 121 | 1360 / 1683 | yes | 112 / 111 |
| Winoground | 800 | 1 | 2 | (not built: gated) | ? | ? | yes (2-way caption choice) | ~100 |

## Where v1 is weak, beyond "can it answer"

1. **Large-option benchmarks (≈15% of public-index weight: BANKING77 + CLINC 7.3%, API-Bank 4.1%, ChessBench 2.2%, POP909 1.7%).**
   - Retrieval & Classification is 20% of the index, and BANKING77 and CLINC150 are gold (1.2×) inside it. Tools is 18.3% of the index, with API-Bank gold. ChessBench and POP909 also need >26 options.
   - Chunked reads are a working fallback, but v1 never saw 77- or 151-label lists, nor a chunk/final format.
   - v1 also excluded the BANKING77 and CLINC train splits. Competing 4B entrants report 0.89 to 0.95 macro-F1 here; they have mostly trained on many-intent data.
2. **Long multi-question rows.**
   - ACOS: 58 yes/no questions on one review. ToolRet and BRIGHT: 32 candidates of up to 31k tokens. ContractNLI: 17 hypotheses on a contract of up to 8k tokens. RouterBench: states of 2k to 5k tokens.
   - v1 trained with states ≤1,536 tokens and typed blocks ≤384 tokens. A 30k-token candidate *inside a question block* is far outside training.
   - The block-bidirectional mask helps here (candidate tokens see each other in the full-attention layers), but the GDN layers read the state before the question.
   - Other entrants gained about 1 public point by "reading twice" (KnowLine Gen5.5) on rows of ≤3k tokens.
3. **Knowledge and reasoning (25.85% of the index)**: MMLU-Pro, GPQA, HLE, BBH, MuSR, CRUXEval, GSM8K. This is bounded by the 4B base. v1's LoRA was not aimed at it, and stock-Qwen3.5-4B harnesses sit at 30 to 37 Full on the board.
4. **Calibration across stages for the fallback**: needs its own held-out fit (see `LARGE_OPTIONS.md`).
5. **Vision.**
   - The images are fine (≤1,600 tokens and ≤3,207 tokens per prefix). The task is multiple choice about charts, documents, multi-view, depth and memes.
   - v1's image training was box, point and count, plus some image choice and noul. Expect roughly base Qwen3.5-4B VL ability, with no OCR or chart-specific tuning.
   - The 1.6 MP regime is 2.5× the token count v1 trained at.
6. **Not a gap**:
   - `noul` (RAGTruth, PhishNChips) maps onto our noul head. Yes/no `choice` maps onto the 2-letter read.
   - No request exceeds our limits, so the `unsupported` count is 0 by design.
   - Option order is never changed.

## Latency risk

The board times 750 rows drawn in proportion to each benchmark's request share. MMLU, MMLU-Pro, HellaSwag, RouterBench and others dominate it; ToolRet gets 5 rows and BRIGHT 3.

On the model above, every per-benchmark **median** is below 1 s except BRIGHT (977 ms). Long-tail rows (ToolRet up to 3.7 s, BRIGHT up to 3.2 s, POP909 up to 1.9 s) move the global mean by only a few ms.

**Estimated gate figures: median ≈ 85 ms, mean ≈ 130 ms, p80 ≈ 165 ms**, a 6× to 10× margin. The real risk is the token-throughput assumption: if our unfused forward ran at only 5k tok/s, the mean would be about 300 ms, still under the gate.

Run `di/latency_probe.py` first on the G4 (see `RUN.md`).
