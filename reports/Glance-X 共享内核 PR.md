# 交给同学的 PR：Glance-X 改用共享内核

给用户的说明（中文）：这份 PR 描述和补丁是给 Glance-X 仓库（`Vision-Decision/Glance-X`，同学维护，公开）准备的，**没有推送，也没有动它的 HF 仓库**。由你决定什么时候、以什么方式交给同学。

- 补丁文件：`reports/glancex_shared_core.patch`（`git am` 可直接应用到 Glance-X `73ee4ed`；作者行是占位符，交出前请改成你的署名）。
- 我们这边已经完成的部分：`dv/glancex/`。它用 dv 的内核和框头加载 Glance-X 的检查点，三种方向的输出与上游逐位相同（见下文"Evidence"）。
- 补丁依赖一个尚不存在的包 `glance-core`。要让补丁可用，我们需要先把 `dv/model/backbone.py` 里的内核部分（第 1–120 行）和 `dv/model/heads.py` 发布成一个小包。建议放在公开的 `Vision-Decision/Glance` 仓库的 `core/` 子目录下，打 tag `core-v0.1`。这一步属于对外发布，需要你拍板。在这之前，补丁只能作为提议交给同学。

---

## PR description (English, for the Glance-X repository)

**Title:** Serve the copied shared-prefix kernels and box heads from one shared package

### Why

`glancex/_layers.py` and `glancex/_box.py` are copies of Glance's `dv/model/backbone.py` (lines 1–120) and
`dv/model/heads.py`. Apart from blank lines and docstrings, `_layers.py` is identical, and `_box.py` is identical once the
Gaussian centre target and `iou_head_on_sel` are removed. Two copies drift: a fix to the fla numerics or to the
left-padding boundary on one side will not reach the other. This PR keeps everything that is new in Glance-X
(the three-direction backbone, the objectness head, data layout, checkpoint format, API) and imports the copied core
from one shared package.

### What changes

| File | Change |
|---|---|
| `glancex/_layers.py` | becomes a re-export of `glance_core.prefix` (`_gdr`, `_gdn_mix`, `_gdn_core`, `gdn_forward`, `_qkv`, `attn_forward`, `build_masks`) |
| `glancex/_box.py` | becomes a re-export of `glance_core.box` (`BoxHeads`, `box_iou_giou`, `center_target_box`, `edge_targets`, `cramer`, `fourier`, `mlp`) |
| `pyproject.toml` | adds `glance-core @ git+https://github.com/Vision-Decision/Glance@core-v0.1#subdirectory=core` |
| `tests/test_core.py` | adds a test that the two modules are thin re-exports, so a second copy cannot creep back in |

No other file changes, and every import site (`backbone.py`, `heads.py`, `train.py`, `evaluate.py`) keeps working
unchanged. Checkpoints are unaffected: `BoxHeads` has the same parameter names, so `heads.safetensors` loads with
`strict=True`.

### Evidence that nothing changes numerically

We ran the reverse direction first: `dv.glancex` (in decision-vision) is Glance-X with exactly this substitution,
with the kernels and heads imported from Glance's code.

1. **Tiny random Qwen3.5, CPU fp32** (`scripts/glancex/test_dv_tiny.py`). For K ∈ {0, 4} and all three directions, the
   readout, the image-token states, every head output, `decode_multi`, the training targets and the loss are
   **bitwise equal** to upstream `73ee4ed` (`outputs/glancex/equivalence_tiny_cpu.txt`, `ALL_OK`).
2. **Released Glance-X-4B checkpoint, CUDA bf16 with the fla kernels** (`scripts/glancex/vm_equivalence.sh`,
   `outputs/glancex/equivalence.json`). Same batches as `glancex.evaluate`, 48 images each of RefCOCO val, gRefCOCO
   testA and the Glance-Data diagnostic set, in all three directions: 5,539 requests per direction (16,617 in total).
   P(present), P(IoU≥.5), the box and all 20 objectness candidates are **bitwise equal**, so no decision differs.
   The same holds on CPU fp32 without fla (`outputs/glancex/equivalence_released_cpu.json`).

### Also offered (separate PRs if wanted)

- **Per-row outputs.** `results/` is git-ignored, so none of the README numbers can be checked offline. We published
  per-request rows for all three directions of the released checkpoint at
  `Vision-Decision/glance-train-ckpts:results/glancex/released/{image_first,text_first,joint}/` (with ids, the
  single box, P(present), P(IoU≥.5), and the top-20 objectness candidates before NMS). Adding a `--rows` flag to
  `glancex/evaluate.py` would make this the default.
- **Official GREC and calibration.** `scripts/glancex/score.py` scores the same rows under the official GREC protocol
  (threshold 0.7, GIoU matching) and reports ECE before and after Platt scaling. Results are in
  `outputs/glancex/summary.md`.
