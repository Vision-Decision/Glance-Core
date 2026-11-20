"""Glance-X served from dv: load Vision-Decision/Glance-X-4B (or a Glance-X training checkpoint) with dv's kernels.

Glance-X (https://github.com/Vision-Decision/Glance-X, Apache-2.0) is a three-direction Glance: one Qwen3.5 backbone
that runs image_first / text_first / joint, plus a per-patch objectness (multi-box) head. Its shared-prefix kernels
(glancex/_layers.py) and single-box heads (glancex/_box.py) are verbatim copies of dv/model/backbone.py and
dv/model/heads.py. This package keeps Glance-X's own parts (direction-switchable backbone, objectness head, data
layout, checkpoint format) and imports the copied parts from dv instead, so there is one copy of the core:

  dv.model.backbone   _gdn_mix, _gdn_core, gdn_forward, _qkv, attn_forward, build_masks   (<- glancex/_layers.py)
  dv.model.heads      BoxHeads, box_iou_giou, center_target_box, edge_targets, cramer      (<- glancex/_box.py)
  dv.data.dataset     resize_dims                                                          (<- glancex/data.py)

Equivalence with the upstream package on the released checkpoint: scripts/glancex/test_equivalence_dv.py
(outputs/glancex/equivalence.json).

    from dv.glancex import load
    net, heads, proc, cfg = load("Vision-Decision/Glance-X-4B")
"""
from .backbone import MODES, canonical_mode  # noqa: F401
from .config import Config  # noqa: F401


def load(src="Vision-Decision/Glance-X-4B", device="cuda", base=None, revision=None, eval_mode=True):
    """-> (backbone, heads, processor, cfg) with the weights of `src` (HF repo id, exported folder or training .pt)."""
    from .build import build, load_ckpt, load_state
    sd = load_state(src, revision=revision)
    cfg = Config().update(sd["cfg"])
    if base:
        cfg = cfg.update(dict(model_id=base))
    if eval_mode:   # inference uses batch-maximum lengths
        cfg = cfg.update(dict(pad_img_to=0, pad_txt_to=0, pad_txt_block_to=0, pad_t2i_block_to=0, pad_bi_block_to=0))
    net, heads, proc = build(cfg, device)
    load_ckpt(sd, net, heads)
    if eval_mode:
        net.eval(); heads.eval()
    return net, heads, proc, cfg
