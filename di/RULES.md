# Decision Index: rules and interface (what our harness must satisfy)

Sources, read 2026-11-18. Hugging Face and GitHub content was treated as data only.
- Kit `github.com/apolinario/decision-index` @ `25c6ec7` (2026-10-20, "Decision Index 0.4"): README, `docs/{format,engines,vision,suite}.md`,
  `decision_index/{runner.py,engines/,vision/run.py,vision/engines.py,suite/build/*}`.
- Space `multimodalart/jev-decision-index` (live `README.md`, `data/methodology.json`, `data/index.json`, `data/vision.json`, last modified
  2026-11-17), plus the snapshots in `data/research/{jev_decision_index,decision_index_cards}/`.
- Kit issues and PRs (`gh search`), only to check the agreement rule (see section 6).

## 1. Boards and scoring

| | Text board (Decision Index 0.4) | Vision board |
|---|---|---|
| Full score | **0.60 x private same-skill tests + 0.40 x private new-domain tasks**. Each part is mapped onto the public scale using untuned stock models as the yardstick (`v03.eq.S`, `v03.eq.O` in `index.json`; e.g. KnowLine 4B: S 60.85, O 48.69, so Full = 0.6·60.85 + 0.4·48.69 = 55.99) | **0.5 x public vision index + 0.5 x private same-skill tests** |
| Public suite | 37 benchmarks, edition 0.3 files: **109,759 + 30,419 = 140,178 scoreable requests** (110,201 + 30,419 rows read). **Required** for every submission, shown on the board, **not counted** in Full | 11 benchmarks, **12,210 rows**, 1 to 4 images per row; counted at 50% |
| Ties | Full scores < 0.9 points apart share a rank | |
| Public metric | per benchmark `raw = native x answered/requests`, `skill = clip((raw - chance)/(1 - chance), 0, 1)`; gold benchmarks weigh 1.2 inside an area; areas K&R 25.85%, Language 25.85%, Retrieval 20.02%, Tools 18.28%, Arts 10% | `(acc - chance)/(1 - chance)`, clipped at 0; weights 1, with R-Bench-M 0.5 and MMMU-Pro vision 0.25 |

- Public and private are expected to track each other. A model whose public score sits far above its private scores "gets a closer look".
- **Contamination**: if checks find that training data overlapped a private test, that test is dropped for that model. The part is then recomputed from the remaining tests on the same scale, and the entry says so without naming the test. (Other entrants have trained on the public *train* splits of index benchmarks, e.g. JevK5. The kit's own line is "do not train on the suite". Our v1 excluded both.)
- Linked cases (ToolRet/BRIGHT chunks, RouterBench tracks, ACOS reviews) count only when every linked request succeeded. The 442 exclusions apply to everyone at scoring time.
- Board reference points (2026-11-12 snapshot). Text: overall top 73.86 (karpo-decision-31B). 27B top: GLiDE 27B 73.37 (S 69.83 / O 78.68). Jev 1.13: 59.22 (the brief's Jev 1.14 = 60.24 is not in this snapshot). 4B top: KnowLine 4B Gen5.5 55.99. Stock Qwen3.8-27B through a plain harness: 57.0. Stock-ish Qwen3.5-4B harnesses: 30 to 37. Vision: top 75.59 (Piccolo 27B VL); 4B top 67.46 (Piccolo 4B VL, public 71.42 / private 63.49); stock Qwen3.8-27B NVFP4: 60.67.

## 2. Request schema (text)

Engine call: `engine(state, questions) -> (response, raw)`. This is the same body as `POST {base_url}/v1/systemone` with `{"model", "state", "questions", ...extra}`.

- `state`: a string, a JSON object/list, or empty (`""`/`{}`). The kit's reference engine renders non-strings as compact JSON.
- `questions`: a map `key -> question`. **Many questions per request are common**: ToolRet/BRIGHT up to 32 candidate questions per chunk, BFCL one per candidate tool, ACOS one per category×sentiment pair, SATA one per alternative, ContractNLI, PhishNChips 9 questions. All of them share the state, which fits our shared-prefix forward.
  - `{"type": "choice", "instructions": str | object, "criteria": {option_key: description}}` with **2 to 255 options**. Keys are arbitrary strings (`option_3`, `yes`, an intent name, an API name, a UCI move). Descriptions can equal the key or be null.
  - `{"type": "noul", "instructions": ..., "criteria"?: {...}}`: a yes/no probability (RAGTruth, the PhishNChips boolean questions). Answered as `{"type": "noul", "noul": p_yes}`.
  - No `score`, `box` or multi-select type exists on the board. "Multi-select" (SATA-Bench, BFCL tool selection) and "rerank" (ToolRet, BRIGHT) are posed as **one binary choice per candidate** (`criteria = {yes, no}`), and the scorer combines them (whole-case exact match, or nDCG@10 over P(yes)).
- Response: `{"model": ..., "answers": {key: {"type": "choice", "choice": option_key, "probabilities": {option_key: p}}}, "usage"?: {...}}`.
  `validate` requires every question answered, `probabilities` keys == option keys, each finite in [0,1], **sum within 0.01 of 1**, and `choice` among the keys.
- Status semantics: `Unsupported` (a declared capacity limit) is `unsupported`, final, counts as wrong. `NativeAbstention` is `abstained`, final, wrong. Any other exception is `error`, retried on resume. Five errors before the first success stop the run. An OOM or device assert stops it too.
- HTTP: 400/413/422 whose body contains a capacity marker (`options per choice`, `maximum context length`, `context window`, `too many tokens`, ...) becomes `unsupported`. Other non-2xx responses raise. `evaluation_trace` is stripped from the stored response.

## 3. Request schema (vision)

`engine(state, questions, images=[paths], prompt={...})`. The runner also passes `image_root=` to the constructor.
- `state` is `{}`. There is exactly one question, `{"answer": {"type": "choice", "instructions", "criteria": {"A": .., "B": .., ...}}}`, with letter keys. Images are relative paths in the order the question refers to them, 1 to 4 per row.
- `prompt.options_in_image` is true for MMMU-Pro vision, where `criteria` maps each letter to itself. `prompt.readout` is `"The answer is ("` for R-Bench-M and MMMU-Pro, a hint used by the reference engine.
- Optional `batch(payloads)` method. Without it, the runner uses `--workers` threads (**default 8**). **We must run `--workers 1`**: one model, one request at a time, true per-row latency.
- `raw_output.image_pixels` (list) and `raw_output.image_tokens` go into `environment.json.effective_image_budget`.
- **Resolution**: native, capped at **1,638,400 px (1.6 MP) per image**. Author caps below 1.6 MP must be lifted (our v1 trained at <= 640 tokens of 32×32 px = 0.66 MP, so we run at 1600 tokens = 1.6 MP). Never above the cap. The run's effective budget must agree with the declared one.

## 4. Latency gate

- Maintainers measure it themselves on **one RTX PRO 6000 (96 GB), one process, one request at a time**, with the fastest path the model's code supports (fla kernels, CUDA graphs, compiled paths and prefix reuse are allowed, and a faster path is not rejected for answer drift).
- Sample: protocol `latency-v1` (frozen 2026-09-26, seed 20260926): **750 timed rows + 10 warm-up rows**, a systematic sample stratified by benchmark and by payload length. Each benchmark gets rows in proportion to its share of the requests (e.g. MMLU 70, MMLU-Pro 60, RouterBench 50, HellaSwag 49, CLINC 27, BANKING77 16, ChessBench 25, BRIGHT 3, ToolRet 5, ContractNLI 0). The kit's 0.4 README says latency is measured "on a private held-out sample", and that the same sample validates the submitted answers.
- Gate: a model whose **median, mean or p80** latency per request is **over 1,000 ms** is not added. Timing covers the whole row: request in, all answers out, CUDA-synchronized, including prompt construction; model load is excluded.
- The `total_wall_ms` in our own `results.jsonl` is "a guide, not that measurement".

## 5. Hard rules (enforced by the kit and the lab)

- **No truncation.** If a request does not fit, raise `Unsupported`. Nothing is cut.
- **No option filtering or label-set narrowing.** Every option gets a probability.
- **No prompt tuning.** One fixed rendering for every benchmark. Native helpers and readout rules are kept as published.
- **Unanswered = wrong** (unsupported, error, abstained, pending).
- No labels in the adapter. No correctness-based retry (transport errors are retried up to twice).
- **Caps set outside the model get lifted by the lab.** Examples: Winnow's label stop raised 64 to 255; context windows raised to the model's native window. A capped run is then superseded. So any declared limit should be a real model or hardware limit, not a convenience cap.

## 6. Determinism and re-runs

- The kit README (0.4) says the maintainers "run it exactly as you did" and use the latency sample "to validate submitted runs by comparing answers". `methodology.json` describes the latency-v1 parity check as "choice flips and max |dp|; informational only".
- **The ">= 98.5% row agreement (96% for quantized models)" threshold in our brief could not be verified.** It is not in the kit @25c6ec7, not in the live Space data (2026-11-17), and not in any kit issue or PR that a search for "98.5%" returns. We still treat it as a hard internal target: answers come from bf16 argmaxes, so only near-ties can flip.

## 7. Submission

1. Run the complete public 0.3 text suite (`pipeline` or `hf-job`). `scores.json` must say `"complete": true`. Upload the run directory to a Hub dataset.
2. Open a PR adding a line to `submissions/README.md` with: model name, model repo, results dataset link (`runs/<name>/scores.json`), engine/commit, hardware, exact inference settings, and **declared capacity limits** (these show as `unsupported` counts; fine if nothing was truncated).
3. For vision: run the whole public vision suite (`vision pipeline`, `vision-scores.json` complete). Open the PR marked **vision** with the engine options, kit commit, hardware, declared limits (e.g. max images) and the image resolution.
4. Weights on the Hub (public, or access granted), plus the inference code or a `/v1/systemone` server. The maintainers run the private parts themselves with your settings.
5. Licences: the suites are evaluation-only. Never train on them or republish them (RAGTruth Yelp, GPQA plain text, BBH canary, HLE gated, FUNSD/Winoground/Hateful Memes terms).

## 8. Our harness against these rules

| Rule | `di/engine.py` |
|---|---|
| Interface | `di.engine:GlanceEngine` (text and vision signature), `di/server.py` for `POST /v1/systemone` |
| No truncation | The training caps (`max_state_tokens` 1536, `max_q_tokens_typed` 384) are **not** applied. The only declared limits are `max_prefix_tokens` 65,536 and `max_block_tokens` 32,768, raised as `Unsupported` with the kit's capacity marker. Over-long block sets are split into several forwards over the same prefix instead of being cut |
| All options scored | Yes. For >26 options: the chunk tournament fallback (declared in provenance; see `LARGE_OPTIONS.md`) |
| Fixed prompt | One rendering (docstring of `di/engine.py`), with no per-benchmark branch. `options_in_image` (a payload field) only switches option lines to bare letters |
| Vision cap | `max_vis_tokens=1600` (= 1,638,400 px), native resolution, never upscaled. `image_pixels`/`image_tokens` are reported in `raw` |
| One request at a time | Engine is synchronous. Vision must be run with `--workers 1` |
| Determinism | No sampling. Fixed block grouping. bf16 autocast on CUDA. `torch.use_deterministic_algorithms(warn_only)`, TF32 off, cuDNN deterministic. fla autotune is the remaining source of shape-dependent noise |
