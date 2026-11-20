"""Box primitive heads: presence gate + center-patch choice + 4 ordinal edge distributions (+ IoU-confidence head).

All geometry is in normalized image coordinates [0,1] (resize-invariant). Visual token (i,j) of an image with
grid (gh, gw) has center ((j+.5)/gw, (i+.5)/gh).
Edge offsets for a chosen patch center (px,py):  left=px-x1, top=py-y1, right=x2-px, bottom=y2-py, each a
distribution over K bins spanning [edge_lo, edge_hi].
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def fourier(xy, n=16):
    freqs = (2.0 ** torch.arange(n, device=xy.device, dtype=xy.dtype)) * math.pi
    a = xy[..., None] * freqs  # [...,2,n]
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)  # [..., 4n]


def box_iou_giou(a, b):
    """a,b: [...,4] xyxy. returns iou, giou."""
    x1 = torch.maximum(a[..., 0], b[..., 0]); y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2]); y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = lambda t: (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)
    union = area(a) + area(b) - inter
    iou = inter / union.clamp(min=1e-9)
    cx1 = torch.minimum(a[..., 0], b[..., 0]); cy1 = torch.minimum(a[..., 1], b[..., 1])
    cx2 = torch.maximum(a[..., 2], b[..., 2]); cy2 = torch.maximum(a[..., 3], b[..., 3])
    c = ((cx2 - cx1) * (cy2 - cy1)).clamp(min=1e-9)
    return iou, iou - (c - union) / c


class BoxHeads(nn.Module):
    def __init__(self, D, cfg):
        super().__init__()
        self.cfg = cfg
        d, K = cfg.head_dim, cfg.edge_bins
        self.K = K
        self.q_norm = nn.LayerNorm(D)
        self.v_norm = nn.LayerNorm(D)
        self.presence = mlp(D, d, 1)
        self.c_q = nn.Linear(D, d)
        self.c_k = nn.Linear(D, d)
        self.edge = mlp(2 * D + 64, 2 * d, 4 * K)
        self.iou = mlp(2 * D + 64, d, 1)
        self.register_buffer("bin_centers", torch.linspace(cfg.edge_lo, cfg.edge_hi, K), persistent=False)
        self.multi = None
        if getattr(cfg, "multibox", False):  # v1.2 multi-instance readout (dv/model/multibox.py); keys "multi.*"
            from .multibox import MultiBoxHeads
            self.multi = MultiBoxHeads(D, cfg)

    # ---------------------------------------------------------------- utils
    def patch_centers(self, gh, gw, device):
        ii, jj = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
        return torch.stack([(jj.flatten() + 0.5) / gw, (ii.flatten() + 0.5) / gh], -1)  # [gh*gw, 2] (x,y)

    def _edge_feats(self, hq, v_sel, pc):
        return torch.cat([hq, v_sel, fourier(pc)], -1)

    def edges_logits(self, hq, v_sel, pc):
        return self.edge(self._edge_feats(hq, v_sel, pc)).view(-1, 4, self.K)

    def expected_box(self, edge_logits, pc):
        p = edge_logits.softmax(-1)
        e = (p * self.bin_centers).sum(-1)  # [M,4]  l,t,r,b
        return torch.stack([pc[:, 0] - e[:, 0], pc[:, 1] - e[:, 1], pc[:, 0] + e[:, 2], pc[:, 1] + e[:, 3]], -1)

    # ---------------------------------------------------------------- forward
    def forward(self, hq, vis, vis_valid, centers, q_img, sel_idx=None, grid_hw=None, multi_k=0):
        """hq:[M,D] question readout; vis:[B,N,D] visual tokens per image (padded); vis_valid:[B,N];
        centers:[B,N,2] patch centers; q_img:[M] image index of each question.
        sel_idx:[M] patch to condition edges on (training: sampled from target). Heads run in fp32.
        grid_hw:[M,2], multi_k>0: also run the multi-instance readout with up to multi_k peak candidates (out["multi"])."""
        hq = self.q_norm(hq.float())
        vis = self.v_norm(vis.float())
        pres = self.presence(hq).squeeze(-1)
        kv = self.c_k(vis)                                    # [B,N,d]
        cl = torch.einsum("md,mnd->mn", self.c_q(hq), kv[q_img]) / math.sqrt(self.c_q.out_features)
        valid = vis_valid[q_img]
        cl = cl.masked_fill(~valid, -1e4)
        pred_idx = cl.argmax(-1)
        out = dict(presence_logit=pres, center_logits=cl, pred_idx=pred_idx)
        v_pred, pc_pred = vis[q_img, pred_idx], centers[q_img, pred_idx]
        el_pred = self.edges_logits(hq, v_pred, pc_pred)
        out["edge_logits_pred"] = el_pred
        out["box_pred"] = self.expected_box(el_pred, pc_pred)
        out["iou_logit"] = self.iou(self._edge_feats(hq, v_pred, pc_pred)).squeeze(-1)
        if sel_idx is not None:
            v_sel, pc_sel = vis[q_img, sel_idx], centers[q_img, sel_idx]
            el = self.edges_logits(hq, v_sel, pc_sel)
            out.update(edge_logits=el, pc_sel=pc_sel, box_sel=self.expected_box(el, pc_sel))
            if getattr(self.cfg, "iou_head_on_sel", False):
                out["iou_logit_sel"] = self.iou(self._edge_feats(hq, v_sel, pc_sel)).squeeze(-1)
        if self.multi is not None and multi_k > 0:
            out["multi"] = self.multi_forward(hq, vis, valid, centers, q_img, grid_hw, multi_k)
        return out

    def multi_forward(self, hq, vis, valid, centers, q_img, grid_hw, k):
        """Peaks of the heat map -> candidate boxes (the shared edge head at each peak), kappa_i, match logits."""
        from .multibox import local_peaks, grad_scale
        mb = self.multi
        gs = getattr(self.cfg, "multi_backbone_grad", 1.0)
        if gs != 1.0:  # damp the new heads' gradient into the shared readout / visual features (and so the LoRA)
            hq, vis = grad_scale(hq, gs), grad_scale(vis, gs)
        if mb.objectness:  # Glance-X: candidates = top-k patches by objectness (NMS at decode), P(match) := objectness
            heat = mb.obj_logits(hq, vis, q_img, valid)
            sc, cidx = heat.detach().topk(min(k, heat.shape[1]), -1)
            cok = valid.gather(1, cidx)
        else:
            heat = mb.heat_logits(hq, vis, q_img, valid)
            cidx, cok = local_peaks(heat.detach(), grid_hw, valid, k)
        M, K = cidx.shape
        v_k, pc_k = vis[q_img[:, None], cidx], centers[q_img[:, None], cidx]           # [M,K,D], [M,K,2]
        hq_k = hq[:, None].expand(M, K, hq.shape[-1])
        with torch.no_grad():  # candidate geometry comes from the (separately trained) single-box edge / IoU heads
            el = self.edges_logits(hq_k.reshape(M * K, -1), v_k.reshape(M * K, -1), pc_k.reshape(M * K, 2))
            cbox = self.expected_box(el, pc_k.reshape(M * K, 2)).view(M, K, 4)
            ciou = self.iou(self._edge_feats(hq_k.reshape(M * K, -1), v_k.reshape(M * K, -1),
                                             pc_k.reshape(M * K, 2))).view(M, K)
        s_k = heat.gather(1, cidx).detach()
        return dict(heat=heat, count_logits=mb.count_logits(hq, heat), cand_idx=cidx, cand_ok=cok, cand_box=cbox,
                    cand_iou_logit=ciou, cand_heat=s_k,
                    cand_match_logit=s_k if mb.objectness else mb.match_logits(hq_k.detach(), v_k.detach(), pc_k, s_k))


# ---------------------------------------------------------------- targets / losses
def center_target(box, centers, valid, grid_hw, cfg):
    if getattr(cfg, "center_target", "box") == "box":
        return center_target_box(box, centers, valid, grid_hw)
    return center_target_gauss(box, centers, valid, grid_hw, cfg)


def center_target_box(box, centers, valid, grid_hw):
    """GUI-Actor style: mass on every patch covered by the GT box, proportional to the covered area fraction."""
    gh, gw = grid_hw[:, 0].float()[:, None], grid_hw[:, 1].float()[:, None]
    hx, hy = 0.5 / gw, 0.5 / gh
    ox = (torch.minimum(centers[..., 0] + hx, box[:, 2:3]) - torch.maximum(centers[..., 0] - hx, box[:, 0:1])).clamp(min=0)
    oy = (torch.minimum(centers[..., 1] + hy, box[:, 3:4]) - torch.maximum(centers[..., 1] - hy, box[:, 1:2])).clamp(min=0)
    t = (ox * oy).masked_fill(~valid, 0)
    t = t / t.sum(-1, keepdim=True).clamp(min=1e-12)
    return t


def center_target_gauss(box, centers, valid, grid_hw, cfg):
    """Gaussian over patches around GT box center. box:[M,4] normalized; centers:[M,N,2]; grid_hw:[M,2] (gh,gw)."""
    gh, gw = grid_hw[:, 0].float(), grid_hw[:, 1].float()
    cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
    sx = ((box[:, 2] - box[:, 0]) * gw * cfg.center_sigma_frac).clamp(cfg.center_sigma_min, cfg.center_sigma_max) / gw
    sy = ((box[:, 3] - box[:, 1]) * gh * cfg.center_sigma_frac).clamp(cfg.center_sigma_min, cfg.center_sigma_max) / gh
    d = ((centers[..., 0] - cx[:, None]) / sx[:, None]) ** 2 + ((centers[..., 1] - cy[:, None]) / sy[:, None]) ** 2
    t = torch.exp(-0.5 * d).masked_fill(~valid, 0)
    # guarantee the patch containing the center has mass even for tiny sigma
    return t / t.sum(-1, keepdim=True).clamp(min=1e-9)


def edge_targets(box, pc, cfg, K):
    offs = torch.stack([pc[:, 0] - box[:, 0], pc[:, 1] - box[:, 1], box[:, 2] - pc[:, 0], box[:, 3] - pc[:, 1]], -1)
    offs = offs.clamp(cfg.edge_lo, cfg.edge_hi)
    bw = (cfg.edge_hi - cfg.edge_lo) / (K - 1)
    centers = torch.linspace(cfg.edge_lo, cfg.edge_hi, K, device=box.device)
    t = torch.exp(-0.5 * ((centers - offs[..., None]) / (cfg.edge_sigma_bins * bw)) ** 2)
    return t / t.sum(-1, keepdim=True), offs


def cramer(logits, target):
    """Squared CDF L2 distance summed over bins, mean over (M,4)."""
    cdf_p = logits.softmax(-1).cumsum(-1)
    cdf_t = target.cumsum(-1)
    return ((cdf_p - cdf_t) ** 2).sum(-1).mean()


def box_loss(out, tgt, cfg):
    present = tgt["present"]
    L = {}
    L["presence"] = F.binary_cross_entropy_with_logits(out["presence_logit"], present.float())
    if present.any():
        pm = present
        ct = tgt["center_t"][pm]
        L["center"] = -(ct * out["center_logits"][pm].log_softmax(-1)).sum(-1).mean()
        et, _ = edge_targets(tgt["box"][pm], out["pc_sel"][pm], cfg, out["edge_logits"].shape[-1])
        L["cramer"] = cramer(out["edge_logits"][pm], et)
        _, giou = box_iou_giou(out["box_sel"][pm], tgt["box"][pm])
        L["giou"] = (1 - giou).mean()
        with torch.no_grad():
            iou_pred, _ = box_iou_giou(out["box_pred"][pm], tgt["box"][pm])
        L["iou_head"] = F.binary_cross_entropy_with_logits(out["iou_logit"][pm], (iou_pred >= 0.5).float())
        L["_acc50"] = (iou_pred >= 0.5).float().mean().detach()
    else:
        z = out["presence_logit"].sum() * 0
        for k in ["center", "cramer", "giou", "iou_head"]:
            L[k] = z
    total = (cfg.w_presence * L["presence"] + cfg.w_center * L["center"] + cfg.w_cramer * L["cramer"]
             + cfg.w_giou * L["giou"] + cfg.w_iou_head * L["iou_head"])
    return total, L


# ---------------------------------------------------------------- v1 additions (point-only targets, masked box loss)
def center_target_points(pt_m, pt_xy, M, centers, valid, grid_hw):
    """Uniform mass over the patches that contain the question's points. pt_m [P] question index, pt_xy [P,2] in [0,1]."""
    t = torch.zeros(M, centers.shape[1], device=centers.device)
    if len(pt_m):
        gh, gw = grid_hw[pt_m, 0], grid_hw[pt_m, 1]
        j = (pt_xy[:, 0] * gw).long().clamp(min=0); j = torch.minimum(j, gw - 1)
        i = (pt_xy[:, 1] * gh).long().clamp(min=0); i = torch.minimum(i, gh - 1)
        t.index_put_((pt_m, i * gw + j), torch.ones(len(pt_m), device=t.device), accumulate=True)
    t = t.masked_fill(~valid, 0)
    return t / t.sum(-1, keepdim=True).clamp(min=1e-12)


def _iou_head_loss(logit, target, cfg):
    l = F.binary_cross_entropy_with_logits(logit, target)
    wb = getattr(cfg, "w_iou_brier", 0.0)
    if wb:
        l = l + wb * ((logit.float().sigmoid() - target) ** 2).mean()
    return l


def box_loss_v1(out, tgt, cfg):
    """Like box_loss, but over box-type questions only; presence on all of them, center on present (box or points),
    edges / GIoU / IoU-head only where a GT box exists. Means are over the respective question subsets."""
    present, has_box, ct = tgt["present"], tgt["has_box"], tgt["center_t"]
    L = {}
    L["presence"] = F.binary_cross_entropy_with_logits(out["presence_logit"], present.float())
    z = out["presence_logit"].sum() * 0
    pc = present & (ct.sum(-1) > 0)
    if "gmask" in tgt and not getattr(cfg, "center_multi", True):
        # v1.2: the centre softmax (-> the single-box answer) is trained on single-referent questions only; "all instances"
        # questions train the heat / match / count heads and the edges. Otherwise a bare noun, which in RefCOCO names ONE
        # instance, learns to spread its mass over all instances (pilot: RefCOCO val 1-word Acc@0.5 96.4 -> 88.8).
        pc = pc & (tgt["gmask"].sum(-1) <= 1)
    L["center"] = -(ct[pc] * out["center_logits"][pc].log_softmax(-1)).sum(-1).mean() if pc.any() else z
    pm = has_box
    if pm.any():
        et, _ = edge_targets(tgt["box"][pm], out["pc_sel"][pm], cfg, out["edge_logits"].shape[-1])
        L["cramer"] = cramer(out["edge_logits"][pm], et)
        _, giou = box_iou_giou(out["box_sel"][pm], tgt["box"][pm])
        L["giou"] = (1 - giou).mean()
        with torch.no_grad():
            if "gt" in tgt:  # several referents: kappa = P(decoded box hits one of them at IoU>=0.5)
                g, gm = tgt["gt"][pm], tgt["gmask"][pm]
                bp = out["box_pred"][pm][:, None].expand_as(g)
                iou_pred = box_iou_giou(bp, g)[0].masked_fill(~gm, 0).max(-1).values
            else:
                iou_pred, _ = box_iou_giou(out["box_pred"][pm], tgt["box"][pm])
        L["iou_head"] = _iou_head_loss(out["iou_logit"][pm], (iou_pred >= 0.5).float(), cfg)
        if "iou_logit_sel" in out:
            with torch.no_grad():
                iou_sel, _ = box_iou_giou(out["box_sel"][pm], tgt["box"][pm])
            L["iou_head"] = 0.5 * (L["iou_head"] + _iou_head_loss(out["iou_logit_sel"][pm], (iou_sel >= 0.5).float(), cfg))
        L["_acc50"] = (iou_pred >= 0.5).float().mean().detach()
    else:
        for k in ["cramer", "giou", "iou_head"]:
            L[k] = z
    total = (cfg.w_presence * L["presence"] + cfg.w_center * L["center"] + cfg.w_cramer * L["cramer"]
             + cfg.w_giou * L["giou"] + cfg.w_iou_head * L["iou_head"])
    return total, L
