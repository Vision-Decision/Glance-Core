# Adapted from Glance-X glancex/train.py (forward_batch, to_dev; commit 73ee4ed, Apache-2.0).
"""One forward in a batch's direction -> head outputs (+ loss targets when training)."""
import torch

from dv.model.heads import center_target_box

from .heads import obj_target, union_center_target


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_batch(net, heads, batch, train=True):
    """Forward in batch['mode']; returns head outputs, loss targets and the per-question geometry."""
    feats = net.image_features(batch["pixel_values"], batch["grid_thw"])
    dev = batch["img_ids"].device.type
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        o = net(batch, feats, batch["mode"])
    q_img = batch["q_img"]
    if batch["mode"] == "i2t":   # vis rows = images
        rows_valid, rows_centers = batch["vis_valid"], batch["centers"]
    else:                        # vis rows = one per question (its own text-conditioned image)
        rows_valid, rows_centers = batch["vis_valid"][q_img], batch["centers"][q_img]
    valid_q, centers_q, grid_q = batch["vis_valid"][q_img], batch["centers"][q_img], batch["grid_hw"][q_img]
    tgt = sel = None
    if train:
        present = batch["present"]
        ct_sel = center_target_box(batch["box_sel"], centers_q, valid_q, grid_q)
        sel = torch.multinomial(ct_sel + (~present)[:, None] * valid_q.float() + 1e-12, 1)[:, 0]
        tgt = dict(present=present, boxes=batch["boxes"], n_boxes=batch["n_boxes"], box_sel=batch["box_sel"],
                   center_t=union_center_target(batch["boxes"], batch["n_boxes"], centers_q, valid_q, grid_q),
                   obj_t=obj_target(batch["boxes"], batch["n_boxes"], centers_q, valid_q, net.cfg.obj_shrink),
                   valid=valid_q)
    out = heads(o["r"], o["vis"], rows_valid, rows_centers, o["vis_index"], sel_idx=sel)
    return out, tgt, (o, rows_valid, rows_centers)
