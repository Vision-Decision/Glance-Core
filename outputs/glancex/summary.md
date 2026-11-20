# Glance-X: reproduction and controls (stopped)

Date: 2026-11-19. Branch `glancex-work`. **Stopped by user decision at ~08:20 UTC: the project focuses on its own work.**
Nothing was pushed to the Glance-X GitHub repo or its HF repos.

## Finished

1. **One copy of the core: `dv/glancex/` serves Glance-X from dv's kernels and box heads.** Glance-X's
   `glancex/_layers.py` and `glancex/_box.py` are copies of `dv/model/backbone.py` and `dv/model/heads.py`.
   `dv.glancex` keeps Glance-X's own parts (three-direction backbone, objectness head, data layout, checkpoint I/O)
   and imports the copied parts from dv. It loads `Vision-Decision/Glance-X-4B` directly.
   - **Bit-exact on the released checkpoint, CUDA bf16 + fla** (`equivalence.json`): 48 images each of RefCOCO val,
     gRefCOCO testA and the Glance-Data diagnostic set, all three directions, **16,617 requests**. Every presence and
     IoU logit, every box and all 20 objectness candidates are identical to upstream `73ee4ed`, so no decision differs.
   - **Also bit-exact on CPU fp32 without fla.** On the released checkpoint: 169 image_first and 8 joint requests
     (`equivalence_released_cpu.json`). On a tiny random Qwen3.5 (K ∈ {0, 4}, all directions): the training targets
     and the loss as well (`equivalence_tiny_cpu.txt`, `ALL_OK`).
2. **Shared-core patch for Glance-X** (not pushed): `reports/glancex_shared_core.patch` and
   `reports/Glance-X 共享内核 PR.md`. The patch turns the two copied files into re-exports of a `glance-core` package.
   That package does not exist yet, and publishing it would be an external release that needs the user's decision.
3. **Glance-X's training mix, rebuilt** from the Glance-Data parquet (`scripts/glancex/prep_data.py`, Glance-Data
   revision `0da8cf88`): 24,407 images, 2,186,531 requests (355,385 original + 1,831,146 Glance-Data), plus a
   ≤1-target view in dv format and the diagnostic set (300 images, 23,578 requests). On HF at
   `Vision-Decision/glance-train-data:glancex/data/` (`data_provenance.json`).
4. **Evaluation tooling** (`scripts/glancex/`):
   - `dump_rows.py`: per-request rows with ids and the objectness candidates before NMS.
   - `score.py`: RefCOCO, gRefCOCO under our definitions and under Glance-X's, official GREC at 0.7, Platt, and the diagnostic set by kind.
   - `paired.py`, `summarize.py`: paired comparisons and summary tables.
   - VM scripts for both control arms.
5. **Partial reproduction:** image_first on the released checkpoint gives Acc@0.5 RefCOCO val / testA / testB =
   **92.86 / 94.87 / 90.44**, against 92.9 / 94.8 / 90.4 in its `docs/RESULTS.md` (rows:
   `Vision-Decision/glance-train-ckpts:glancex/partial/gx-b/rows/released/image_first/`).

## Stopped (partial artifacts on HF, labelled `PARTIAL.md`)

| Job | Reached | HF |
|---|---|---|
| Released-checkpoint eval, 3 directions × 12 sets | image_first: 3 of 12 sets; the other two directions not started | `glancex/partial/gx-b/rows/` |
| Control arm B: Glance-X trainer, `image_first` only, seed 0 | step 380 / 6,100, no checkpoint | `glancex/partial/gx-b/train/i2t_only_s0/` (+ `glancex/train/i2t_only_s0/` logs) |
| Control arm B, seed 1 | step ~260 / 6,100, no checkpoint | `glancex/partial/gx-c/train/i2t_only_s1/` |
| Arm C: our recipe on Glance-X's mix | step ~230 / 6,100; checkpoint snapshot at step 159 (not a usable model) | `glancex/partial/gx-c/train/dvmix_s0/` |

Not done, so open as before: the data vs. three-direction attribution of the single-target gap (90.9 vs 92.8);
official GREC, calibration (ECE raw / Platt) and the multi-target F1 of Glance-X; the diagnostic set by kind. The
pipeline runs end to end, so resuming needs only GPU time. One G4 per arm comes to about 6–9 h each at the measured
3.8–5.3 s/step; the eval is a few hours per direction. VMs gx-b and gx-c are stopped and deregistered.
