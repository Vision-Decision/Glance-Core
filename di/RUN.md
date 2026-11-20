# GPU run plan: Glance v1 on the full public Decision Index suites (one Colab G4)

**Status: not launched.** All Colab GPU slots were busy on 2026-11-18. Everything below is ready to paste. Run it from
the repository root on the Mac. Session name `dv-di`, G4 = RTX PRO 6000 Blackwell 96 GB, the board's latency hardware.

## Budget

| stage | requests | estimate on G4 |
|---|---:|---|
| setup (pip, Qwen3.5-4B weights, release adapter + heads ~0.5 GB from HF) | | 0.2 h |
| **build_text**: rebuild the text suite on the VM from pinned sources (~7 GB down, ~17 GB disk) | | 0.7 h (0.5 to 1.2 h; network-bound) |
| latency probe (latency-v1 style, 750 + 10 rows) | 760 | 0.05 h |
| determinism check (3,000-row stratified sample, run twice in fresh processes) | 6,000 | 0.25 h |
| **text suite 0.3** | 140,178 scoreable (140,620 run) | **5.0 h** (model: 70 ms/request + 140 M tokens / 20k tok/s); range 4.2 to 7 h if throughput is 40k to 10k tok/s |
| **build_vision**: rebuild the vision suite on the VM (~5.7 GB down) | | 0.4 h |
| **vision suite** | 12,210 | **0.35 h** |
| scoring, download, slack | | 0.3 h |
| **total** | ~159k | **≈ 7.3 h (range 6 to 10 h)** |

At the G4 rate seen for v1 (≈ 10 CU/h, `outputs/v1/run_summary.json`): **≈ 73 CU expected, 60 to 100 CU range.**
The probe recalibrates this within 5 minutes. If its p50 is above 150 ms or its mean is above 400 ms, stop and re-plan
before the full suite.

Where the tokens go: RouterBench 26 M, ToolRet 22 M, POP909 13 M, ACOS 10 M and BRIGHT 7 M make up more than half
the text tokens. `di/suite_stats.json` has the rest.

## Data policy

- **No bulk data on or from the laptop.** The only uploads are the code tarball (`dv/ di/ eval.py configs/`, < 1 MB)
  and the kit tarball (`data/di/kit-25c6ec7.tgz`, 1.3 MB).
- **Weights** come from the public release **`Vision-Decision/Glance-4B-preview`**, pinned to revision `b6237b4`. That is v1:
  step 3844, the same temperatures as `outputs/v1/eval_results.json`. The release path and the local-checkpoint path
  were checked to agree to about 1e-9.
- **Later versions**: `MODEL=<private training-checkpoint repo> REVISION=<sha>` on the VM. The repo needs the same
  three files (`glance_config.json`, `adapter_model.safetensors`, `heads.safetensors`) and an HF token with read
  access.
- **Suites are rebuilt on the VM, not mirrored.** I did not create a private HF cache (e.g. `Vision-Decision/di-suite-cache`)
  because several sources do not allow mirroring, even privately:
  - RAGTruth's Yelp contexts: no redistribution;
  - Hateful Memes: no redistribution of derivatives;
  - RealWorldQA: CC BY-ND;
  - Winoground: research-only agreement;
  - FUNSD: non-commercial;
  - GPQA: asks that its questions not be posted.

  The kit's README suggests a private dataset for HF Jobs, but the source terms win. The rebuild costs about 1.1 h
  of G4 time per VM; that is the price.
- What stays local: `di/suite_stats.json` (the measured shapes) and `data/di/` with the kit tgz and the suite
  manifests/hashes only.

## Manual steps for the user (before launch)

1. **Winoground** (vision, 800 rows) is gated. Accept its terms yourself at
   https://huggingface.co/datasets/facebook/winoground, with the HF account whose token the VM will use. I did not
   accept them. Without this, `build_vision` stops with "WINOGROUND NOT BUILT" and the vision run cannot be complete.
2. **HF token on the VM**: HLE (text) and Winoground are gated, and the VM needs a read token from an account that has
   access. `hf auth whoami` on the laptop is `user`, which already passed HLE's gate on 2026-11-18. Put the token on
   the VM your usual way, e.g. a Colab secret, or `huggingface-cli login` inside `scripts/rsh.sh`. `setup` checks it
   with `whoami()` and fails early otherwise.

## Launch

```bash
S=dv-di
colab-pool new $S --gpu g4                      # exit code 69 = no capacity; wait, don't switch accounts

# the only uploads: code (< 1 MB) and the kit (1.3 MB)
COPYFILE_DISABLE=1 tar czf /tmp/dv-di-code.tgz --exclude='__pycache__' --exclude='smoke_results*' dv di eval.py configs
colab-pool upload $S /tmp/dv-di-code.tgz /content/dv.tgz
colab-pool upload $S data/di/kit-25c6ec7.tgz /content/kit.tgz
scripts/rsh.sh $S "mkdir -p /content/dv && tar xzf /content/dv.tgz -C /content/dv && ls /content/dv"
# (manual step 2: HF token on the VM)

# stage 0: setup (foreground, ~10 min): deps, kit, Qwen3.5-4B, Glance-4B-preview @b6237b4, token check
RSH_TIMEOUT=3000 RSH_WALL=3000 scripts/rsh.sh $S "bash /content/dv/di/colab_run.sh setup > /content/di_setup.log 2>&1; tail -15 /content/di_setup.log"

# stage 1: rebuild the text suite on the VM (detached, ~40-60 min), then the latency probe
CMD="{ bash /content/dv/di/colab_run.sh build_text && bash /content/dv/di/colab_run.sh probe; }" \
LOG=/content/out/di_build_probe.log python3 scripts/remote_bg.py | colab-pool exec $S
scripts/rsh.sh $S "tail -5 /content/out/di_build_probe.log; cat /content/out/di/latency_probe.json"   # read before going on

# stages 2 to 5, detached: determinism, text suite, vision rebuild, vision suite (all resumable)
CMD="{ bash /content/dv/di/colab_run.sh agree && bash /content/dv/di/colab_run.sh text && bash /content/dv/di/colab_run.sh build_vision && bash /content/dv/di/colab_run.sh vision; }" \
LOG=/content/out/di_run.log python3 scripts/remote_bg.py | colab-pool exec $S
```

`scripts/remote_bg.py` appends `> LOG 2>&1` to CMD, so a chain must be grouped in `{ ...; }` or only its last command is logged.
Text and vision can run on two VMs in parallel: put `build_vision` + `vision` on the second one; they need no text suite.

### Multi-VM (what the 2026-11-18 run did)

The text suite shards cleanly. Each row is still timed alone on one G4, so the per-row latency is unchanged.

- **VM roles**:
  - `dv-di-text` (acct-0): `setup`, `build_text`, `probe`, `agree`, then shard 0.
  - `dv-di-vis` (acct-7): `setup`, `build_vision`, `vision`, then `build_text` and shard 1.
  - `dv-di-text2` (acct-0): `setup`, `build_text`, shard 2.
- **Shard selection**: `/content/di/shard.conf` = `K 0,7,14,20` makes `colab_run.sh text` run only shard K. A row's
  shard is sha256(group_id) mod 20, so linked cases stay together.
- **Merging**: shards 1 and 2 are stripped of `payload` (model outputs only, no suite text) with `di/shard.py strip`
  and pushed to the private dataset `Vision-Decision/di-runs`. On `dv-di-text`, download them into `/content/di/shard_results/`
  and run `colab_run.sh score_text` (merge plus the kit's `score`).
- **Timing**: the probe measured BRIGHT at a 9.0 s median and ToolRet at 6.8 s, against about 1 s in the
  `suite_stats` model. That puts one VM at about 7 h for text; three shards take about 2.5 h each.
- **Gotchas**:
  - A `remote_bg.py` chain must be grouped in `{ ...; }`.
  - Never combine `curl --retry` with `-C -`: the HoVer DB came out 22 MB too long.

### Suite integrity

- On macOS (2026-11-18) the rebuilt `added-rows` and `gsm8k-rows` matched the lab's hashes. `selected-rows` differed
  **only through SGD**: the normalized `SGD-service-given-intent.jsonl` was ours `e57d11fd…`, the lab's `8496e556…`,
  with the same 2,500 run ids. SGD is shown but not counted.
- `build_text` therefore always finishes through `di/finish_suite.py`. That script imports with `--no-verify` only if
  the rows don't match.
- If the Linux rebuild matches, set `NOVERIFY=0`. If it doesn't, note it in the PR or ask the maintainers on the kit's
  tracker.
- Vision: InfographicVQA and CORD digests need Pillow 12.3.0 (pinned in `setup`).
- Mind2Web digests depend on the Arial Bold font build. The VM uses `ttf-mscorefonts-installer`. If
  `vision_verify.json` flags Mind2Web, the boxes are identical and only the label glyphs differ; say so in the PR.

### Monitor, pull, stop

```bash
scripts/rsh.sh dv-di "cat /content/out/di/agreement.json; tail -c 600 /content/out/di/text-glance-v1/status.json; tail -3 /content/out/di_run.log"
# keepalive while waiting (every 5 min):  scripts/rsh.sh dv-di "echo ok"
mkdir -p outputs/di && for f in latency_probe.json agreement.json; do colab-pool download dv-di /content/out/di/$f outputs/di/$f; done
scripts/rsh.sh dv-di "cd /content/out/di && tar czf /content/di_runs.tgz text-glance-v1 vision-glance-v1"
colab-pool download dv-di /content/di_runs.tgz outputs/di/di_runs.tgz && tar xzf outputs/di/di_runs.tgz -C outputs/di
colab-pool stop dv-di
```

If the VM dies, run `colab-pool new` again, upload again, and re-run the same stage. `results.jsonl` resumes and
retries only `error` rows. Download `results.jsonl` periodically to keep a copy.

## Pass criteria before anything is submitted

- `latency_probe.json`: `gate_ok` true. Expected median ≈ 85 ms, mean ≈ 130 ms, p80 ≈ 165 ms.
- `agreement.json`: agreement ≥ 0.985 (our internal target; the threshold isn't in the public rules, see `RULES.md` §6).
- `text-glance-v1/scores.json`: `"complete": true`, 0 errors, 0 unsupported expected.
  `vision-glance-v1/vision-scores.json`: complete. `environment.json` `effective_image_budget.max_image_pixels` ≤ 1,638,400.
- Upload to a **private** Hub dataset and open the submissions PR only after a human decision. This is not part of
  this plan.

## Settings to quote in a submission

`--engine di.engine:GlanceEngine` (weights `Vision-Decision/Glance-4B-preview` @b6237b4, the release's temperatures) with these defaults:
- bf16 autocast on CUDA, `max_vis_tokens=1600` (1.6 MP);
- `chunk_mode=tournament`, `advance=3`;
- `max_prefix_tokens=65536`, `max_block_tokens=32768`, `group_tokens=32768`;
- kit @25c6ec7, transformers 5.23.0, flash-linear-attention 0.6.0, one RTX PRO 6000;
- vision with `--workers 1`.

Declared limits: 255 options; 65,536-token prefix; 32,768-token block. None is hit on the public suite.

## Local smoke test (done 2026-11-18, this Mac)

Run on MPS in fp32. CPU fp32 gives the same probabilities to about 1e-8 but takes about 35 s per request; MPS takes
0.5 to 16 s.

- `di/smoke_test.py --per_bench 2 --max_chars 6000`: **101/101 requests valid**, 0 errors, 0 unsupported. It covered
  all 45 text and vision benchmarks it could sample (Winoground was not built), plus a synthetic 77-option row.
  - Batch tensors are identical to `dv.data.mixed.collate_mixed` (text record and image record).
  - 18 repeated requests gave identical answers (max |dp| = 0).
  - Results: `di/smoke_results_summary.json`. These files hold gold labels of public items: keep them local and
    don't commit them.
- Long and large-option rows, run separately:
  - API-Bank (6.7k-token catalog, 53 options, 3 chunks): 2/2 correct.
  - POP909 (129 chords, 5 chunks): 0/2 correct.
  - ToolRet (32 candidates, 2 grouped forwards): runs.
  - BRIGHT (32 candidates): runs.
  - ContractNLI (17 questions on a 3.1k-token contract): 13/17 correct.
  - ACOS (64 questions): 64/64 correct.
  - CLINC150 (151 options): 4/4 correct. BANKING77: 2/4 correct. ChessBench: 0/4 correct.

  The per-benchmark samples are tiny (n ≤ 4): they show the plumbing works, not accuracy.
- `di/server.py` behind the kit's `http` engine: valid responses. A 300-option request comes back as `Unsupported`
  ("options per choice").
