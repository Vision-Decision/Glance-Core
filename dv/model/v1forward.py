"""v1 forward over a mixed batch (image and/or text prefixes; box + typed questions). Shared by train / eval / inference."""
import torch

from .heads import center_target_box, center_target_points
from .typed import typed_logits


def pad_gt(gb_m, gb_xyxy, bi, M):
    """flat referent boxes (gb_m = question index in the batch) -> padded [len(bi), G, 4] + mask for the box questions bi"""
    remap = torch.full((M,), -1, dtype=torch.long, device=bi.device)
    remap[bi] = torch.arange(len(bi), device=bi.device)
    m = remap[gb_m] if len(gb_m) else gb_m
    keep = m >= 0
    m, xy = m[keep], gb_xyxy[keep]
    Mb = len(bi)
    cnt = torch.bincount(m, minlength=Mb) if len(m) else torch.zeros(Mb, dtype=torch.long, device=bi.device)
    G = max(1, int(cnt.max())) if Mb else 1
    gt = xy.new_zeros(Mb, G, 4)
    gmask = torch.zeros(Mb, G, dtype=torch.bool, device=bi.device)
    if len(m):
        order = torch.argsort(m, stable=True)
        m, xy = m[order], xy[order]
        start = torch.cumsum(cnt, 0) - cnt
        slot = torch.arange(len(m), device=m.device) - start[m]
        gt[m, slot] = xy
        gmask[m, slot] = True
    return gt, gmask


def multi_center_target(gt, gmask, cen, valid, ghw):
    """Centre-softmax target for 1..N referents: each referent's GUI-Actor coverage target (normalized), averaged with equal
    weight per referent (= the old single-box target when N = 1). Also returns the per-referent coverage [Mb,G,N]."""
    Mb, G = gmask.shape
    N = cen.shape[1]
    cov = center_target_box(gt.reshape(Mb * G, 4), cen.repeat_interleave(G, 0), valid.repeat_interleave(G, 0),
                            ghw.repeat_interleave(G, 0)).view(Mb, G, N) * gmask[..., None]
    ct = cov.sum(1) / gmask.sum(1, keepdim=True).clamp(min=1)
    return ct, cov


def run_model_v1(net, heads, batch, train=True):
    """Mixed batch: image and/or text prefixes; box + typed questions. Returns dict of head outputs."""
    frozen = getattr(heads.box.cfg, "train_only", "") == "multi"  # v1.2 add-on training: backbone is a fixed feature map
    with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
        feats = net.image_features(batch["pixel_values"], batch["grid_thw"]) if batch["pixel_values"] is not None else []
        if batch.get("pixel_values_videos") is not None:  # Glance-Video: batches are homogeneous (sampler groups by kind)
            assert not feats, "mixed image+video micro-batch"
            feats = net.video_features(batch["pixel_values_videos"], batch["video_grid_thw"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hp, hq = net(batch, feats)
    M = hq.shape[0]
    hq_last = hq[torch.arange(M, device=hq.device), batch["q_last"]]
    res = dict(qtype=batch["qtype"])
    is_box = batch["qtype"] == 0
    typed = ~is_box
    if typed.any():
        lab = heads.typed(hq_last[typed])
        res["typed_logits"] = typed_logits(lab, batch["lab_idx"][typed])
        res["typed_mask"] = typed
    if is_box.any():
        bi = is_box.nonzero()[:, 0]
        B, N = batch["vis_valid"].shape
        vis = hp.new_zeros(B, N, hp.shape[-1])
        vis[batch["vis_valid"]] = hp[batch["p_img_mask"]]
        q_img = batch["q_img"][bi]
        valid = batch["vis_valid"][q_img]
        cen = batch["centers"][q_img]
        ghw = batch["grid_hw"][bi]
        ct = sel = None
        multi = getattr(heads.box, "multi", None) is not None
        if "gb_m" in batch:
            gt, gmask = pad_gt(batch["gb_m"], batch["gb_xyxy"], bi, M)
        else:
            gt, gmask = batch["box"][bi][:, None], batch["has_box"][bi][:, None]
        if train:
            ct, cov = multi_center_target(gt, gmask, cen, valid, ghw)
            if len(batch["pt_m"]):
                remap = torch.full((M,), -1, dtype=torch.long, device=bi.device)
                remap[bi] = torch.arange(len(bi), device=bi.device)
                pm = remap[batch["pt_m"]]
                keep = pm >= 0
                ct = ct + center_target_points(pm[keep], batch["pt_xy"][keep], len(bi), cen, valid, ghw)
            noct = ct.sum(-1) <= 0
            sel = torch.multinomial(ct + noct[:, None] * valid.float() + 1e-12, 1)[:, 0]
            # edges / GIoU are supervised with the referent whose coverage at the sampled patch is largest
            a = cov.gather(2, sel[:, None, None].expand(-1, cov.shape[1], 1))[..., 0].masked_fill(~gmask, -1).argmax(1)
            res["box_t"] = gt[torch.arange(len(bi), device=bi.device), a]
        k = (cfg_k(heads, train)) if multi else 0
        out = heads.box(hq_last[bi], vis, batch["vis_valid"], batch["centers"], q_img, sel_idx=sel, grid_hw=ghw, multi_k=k)
        res.update(box=out, box_idx=bi, center_t=ct, gt=gt, gmask=gmask)
        if train and multi:
            from .multibox import heat_target
            pm = None
            if len(batch["pt_m"]):
                remap = torch.full((M,), -1, dtype=torch.long, device=bi.device)
                remap[bi] = torch.arange(len(bi), device=bi.device)
                pm = remap[batch["pt_m"]]
                keep = pm >= 0
                pm, pxy = pm[keep], batch["pt_xy"][keep]
            hc = heads.box.cfg
            Y, pos = heat_target(gt, gmask, cen, valid, ghw, hc.heat_sigma_frac, hc.center_sigma_min,
                                 pt_m=pm, pt_xy=pxy if pm is not None else None)
            n = batch["n_tgt"][bi] if "n_tgt" in batch else gmask.sum(1)
            res["multi_t"] = dict(present=batch["present"][bi], gt=gt, gmask=gmask, Y=Y, pos=pos, valid=valid,
                                  n=n, n_known=n > 0, centers=cen)
    return res


def cfg_k(heads, train):
    c = heads.box.cfg
    return c.multi_k_train if train else c.multi_k
