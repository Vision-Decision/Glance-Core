"""Multi-instance box readout (v1.2): 0..N boxes from the same <|box_start|> readout, one prefill, no decoding.

The single-box heads (presence gate g, centre softmax, 4 edge distributions, IoU head kappa) are kept unchanged and still
give the single-box answer. This module adds, on the same features:

  heat   : per-patch sigmoid  q_j = P(patch j holds the centre of a referent | present)   (CenterNet-style focal loss)
  peaks  : local maxima of the heat logits on the patch grid (3x3, no NMS threshold to tune), top-K
  boxes  : the existing edge head read at every peak patch (the edge head is patch-conditioned, so a peak on instance i
           decodes instance i's box); kappa_i from the existing IoU head at the same patch
  match  : per-candidate m_i = P(candidate i is a distinct correct box | present), trained with a set-style label: candidates
           are visited in heat order and matched one-to-one to unmatched GT boxes at IoU>=0.5 (greedy, as in detection AP).
           A second peak on an already-matched instance gets label 0, so the head learns de-duplication; BCE is a proper
           scoring rule, so m_i is calibrated (the per-box confidence is g * m_i)
  count  : P(n=1,2,3,>=4 | present) from the readout (+ log of the summed heat); P(n=0) = 1 - g, i.e. the presence gate
           stays the one "not here" knob and the count is a distribution over the remaining mass

Decoding (decode_sets): boxes = {candidate i : m_i >= thr} if g >= 0.5, else {}; if the gate says present but no candidate
passes, the single-box answer is returned so the set API never contradicts the single API.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import mlp, fourier, box_iou_giou

def grad_scale(x, s):
    """identity forward, gradient * s backward"""
    return x * s + x.detach() * (1 - s)


NEI = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class MultiBoxHeads(nn.Module):
    def __init__(self, D, cfg):
        super().__init__()
        d = cfg.head_dim
        self.h_q = nn.Linear(D, d)
        self.h_k = nn.Linear(D, d)
        self.h_scale = nn.Parameter(torch.tensor(1.0))
        self.h_off = nn.Linear(D, 1)
        nn.init.zeros_(self.h_off.weight); nn.init.zeros_(self.h_off.bias)
        self.match = mlp(2 * D + 64 + 1, d, 1)
        self.count = mlp(D + 1, d, cfg.count_classes)
        self.objectness = getattr(cfg, "multi_head", "heat") == "objectness"
        if self.objectness:  # Glance-X objectness head: its own bilinear, prior bias -4 (most patches are not centres)
            self.o_q = nn.Linear(D, d)
            self.o_k = nn.Linear(D, d)
            self.o_b = nn.Parameter(torch.tensor(-4.0))

    def obj_logits(self, hq, vis, q_img, valid):
        ol = torch.einsum("md,mnd->mn", self.o_q(hq), self.o_k(vis)[q_img]) / math.sqrt(self.o_q.out_features) + self.o_b
        return ol.masked_fill(~valid, -30)

    @torch.no_grad()
    def init_from(self, box):
        """Start the heat bilinear from the trained centre-softmax bilinear: the heat peak then sits where the softmax peak
        is, at logit h_off(h) = 0 (q = 0.5), and every other patch below it."""
        self.h_q.load_state_dict(box.c_q.state_dict())
        self.h_k.load_state_dict(box.c_k.state_dict())

    def heat_logits(self, hq, vis, q_img, valid):
        """hq [M,D], vis [B,N,D] (both already layer-normed), valid [M,N] -> [M,N]"""
        dot = torch.einsum("md,mnd->mn", self.h_q(hq), self.h_k(vis)[q_img]) / math.sqrt(self.h_q.out_features)
        dot = dot.masked_fill(~valid, -1e4)
        mx = dot.max(-1, keepdim=True).values.detach()
        # h_off is a 1/sqrt(D)-scaled linear: with Adam, an unscaled D->1 map moves the logit by ~lr * sum|h| (~2 at
        # lr 1e-3, D = 2560) per step, which made the heat loss diverge in the first pilot. Logits are soft-clamped to (-30, 30).
        s = self.h_scale * (dot - mx) + self.h_off(hq) / math.sqrt(hq.shape[-1])
        return (30 * torch.tanh(s / 30)).masked_fill(~valid, -30)

    def count_logits(self, hq, heat):
        mass = torch.log1p(heat.sigmoid().sum(-1, keepdim=True))
        return self.count(torch.cat([hq, mass], -1))

    def match_logits(self, hq_k, v_k, pc_k, s_k):
        return self.match(torch.cat([hq_k, v_k, fourier(pc_k), s_k[..., None]], -1)).squeeze(-1)


def local_peaks(s, grid_hw, valid, k):
    """Local maxima (8-neighbourhood, ties broken towards the lower index) of per-patch logits s [M,N] on each question's
    (gh,gw) grid; patch j sits at row j // gw, col j % gw. Returns idx [M,k'] sorted by logit, ok [M,k'] (k' = min(k,N))."""
    M, N = s.shape
    gh, gw = grid_hw[:, 0:1].long(), grid_hw[:, 1:2].long().clamp(min=1)
    j = torch.arange(N, device=s.device)[None].expand(M, N)
    r, c = j // gw, j % gw
    peak = valid.clone()
    for dr, dc in NEI:
        rr, cc = r + dr, c + dc
        inb = (rr >= 0) & (rr < gh) & (cc >= 0) & (cc < gw)
        nj = (rr * gw + cc).clamp(0, N - 1)
        ns = s.gather(1, nj)
        beats = inb & ((ns > s) | ((ns == s) & (nj < j)))
        peak &= ~beats
    sc = s.masked_fill(~peak, float("-inf"))
    v, idx = sc.topk(min(k, N), -1)
    return idx, torch.isfinite(v)


def heat_target(gt, gmask, centers, valid, grid_hw, sigma_frac=0.15, sigma_min=0.5, pt_m=None, pt_xy=None):
    """CenterNet-style target. gt [M,G,4] normalized xyxy, gmask [M,G]; centers [M,N,2]. Each GT box contributes a Gaussian
    (sigma = sigma_frac * box size in patches, >= sigma_min) around its centre; Y = max over boxes; the patch holding each
    centre is a positive (Y = 1). Optional points (pt_m [P] question index, pt_xy [P,2]) are positives with sigma_min.
    Returns Y [M,N] in [0,1] and pos [M,N] bool."""
    M, N = centers.shape[:2]
    gh, gw = grid_hw[:, 0].float(), grid_hw[:, 1].float()
    Y = centers.new_zeros(M, N)
    pos = torch.zeros(M, N, dtype=torch.bool, device=centers.device)

    def splat(m_idx, cx, cy, sx, sy):
        c = centers[m_idx]                                            # [P,N,2]
        d = ((c[..., 0] - cx[:, None]) / sx[:, None]) ** 2 + ((c[..., 1] - cy[:, None]) / sy[:, None]) ** 2
        g = torch.exp(-0.5 * d).masked_fill(~valid[m_idx], 0)
        Y.index_reduce_(0, m_idx, g, "amax")
        jj = (cx * gw[m_idx]).long().clamp(min=0); jj = torch.minimum(jj, gw[m_idx].long() - 1)
        ii = (cy * gh[m_idx]).long().clamp(min=0); ii = torch.minimum(ii, gh[m_idx].long() - 1)
        pos[m_idx, ii * gw[m_idx].long() + jj] = True

    if gmask.any():
        m_idx, g_idx = gmask.nonzero(as_tuple=True)
        b = gt[m_idx, g_idx]
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        sx = ((b[:, 2] - b[:, 0]) * gw[m_idx] * sigma_frac).clamp(min=sigma_min) / gw[m_idx]
        sy = ((b[:, 3] - b[:, 1]) * gh[m_idx] * sigma_frac).clamp(min=sigma_min) / gh[m_idx]
        splat(m_idx, cx, cy, sx, sy)
    if pt_m is not None and len(pt_m):
        splat(pt_m, pt_xy[:, 0], pt_xy[:, 1], sigma_min / gw[pt_m], sigma_min / gh[pt_m])
    pos &= valid
    Y = torch.where(pos, torch.ones_like(Y), Y).masked_fill(~valid, 0)
    return Y, pos


def focal_heat_loss(s, Y, pos, valid, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss, summed over patches / #positives, per question. s [M,N] logits. -> [M]"""
    p = s.sigmoid().clamp(1e-6, 1 - 1e-6)
    lp, ln = F.logsigmoid(s), F.logsigmoid(-s)
    lpos = -((1 - p) ** alpha) * lp
    lneg = -((1 - Y) ** beta) * (p ** alpha) * ln
    l = torch.where(pos, lpos, lneg).masked_fill(~valid, 0).sum(-1)
    return l / pos.sum(-1).clamp(min=1)


@torch.no_grad()
def greedy_match_labels(cand_box, cand_ok, gt, gmask, thr=0.5):
    """cand_box [M,K,4] in heat order, gt [M,G,4]. Candidate k gets label 1 iff its best IoU among still-unmatched GT boxes
    is >= thr (that GT is then consumed). -> labels [M,K] float, best IoU [M,K]"""
    M, K = cand_box.shape[:2]
    G = gt.shape[1]
    lab = cand_box.new_zeros(M, K)
    best = cand_box.new_zeros(M, K)
    if G == 0:
        return lab, best
    iou, _ = box_iou_giou(cand_box[:, :, None, :].expand(M, K, G, 4), gt[:, None].expand(M, K, G, 4))  # [M,K,G]
    iou = iou.masked_fill(~gmask[:, None], -1)
    free = gmask.clone()
    for k in range(K):
        x = iou[:, k].masked_fill(~free, -1)
        v, g = x.max(-1)
        hit = cand_ok[:, k] & (v >= thr)
        lab[:, k] = hit.float()
        best[:, k] = iou[:, k].max(-1).values.clamp(min=0)
        free[hit, g[hit]] = False
    return lab, best


def obj_target(gt, gmask, centers, valid, shrink=0.5):
    """Glance-X objectness target: a patch is positive if its centre lies in the central `shrink` part of any referent box;
    every referent keeps at least its nearest patch. gt [M,G,4], gmask [M,G], centers [M,N,2] -> [M,N] float"""
    M, G = gmask.shape
    t = centers.new_zeros(M, centers.shape[1])
    for j in range(G):
        has = gmask[:, j]
        if not has.any():
            continue
        b = gt[has, j]
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        hw, hh = (b[:, 2] - b[:, 0]) * shrink / 2, (b[:, 3] - b[:, 1]) * shrink / 2
        c = centers[has]
        inside = ((c[..., 0] - cx[:, None]).abs() <= hw[:, None]) & ((c[..., 1] - cy[:, None]).abs() <= hh[:, None])
        d = ((c[..., 0] - cx[:, None]) ** 2 + (c[..., 1] - cy[:, None]) ** 2).masked_fill(~valid[has], 1e9)
        near = F.one_hot(d.argmin(-1), c.shape[1]).bool()
        t[has] = torch.maximum(t[has], (inside | near).float())
    return t * valid.float()


def sigmoid_focal(logits, target, valid, alpha=0.25, gamma=2.0):
    """Glance-X objectness loss (RetinaNet focal, summed over patches / #positives, mean over questions)"""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    loss = (w * (1 - pt) ** gamma * ce) * valid.float()
    return (loss.sum(-1) / target.sum(-1).clamp(min=1.0)).mean()


def multi_loss(mo, tgt, cfg):
    """mo: multi outputs (heat, count_logits, cand_*); tgt: present [M], gt [M,G,4], gmask [M,G], Y/pos [M,N], valid,
    n_known [M] (count known), n [M]. Heat / count / match losses on present questions only (conditional on presence)."""
    L = {}
    z = mo["heat"].sum() * 0
    present = tgt["present"]
    if getattr(cfg, "multi_head", "heat") == "objectness":
        # Glance-X: objectness on every box question (absent ones -> all-negative target); no match / count heads
        ot = obj_target(tgt["gt"], tgt["gmask"] & present[:, None], tgt["centers"], tgt["valid"], getattr(cfg, "obj_shrink", 0.5))
        L["heat"] = sigmoid_focal(mo["heat"], ot, tgt["valid"])
        with torch.no_grad():
            p = mo["heat"].sigmoid(); pos = ot > 0; neg = tgt["valid"] & ~pos
            L["_heat_p_neg"] = (p * neg).sum() / neg.sum().clamp(min=1)
            L["_heat_p_pos"] = (p * pos).sum() / pos.sum().clamp(min=1)
        L["count"] = z; L["match"] = z
        return cfg.w_heat * L["heat"], L
    hm = present & tgt["pos"].any(-1)
    if hm.any():
        hl = focal_heat_loss(mo["heat"][hm], tgt["Y"][hm], tgt["pos"][hm], tgt["valid"][hm])
        if getattr(cfg, "heat_loss_norm", "pos") == "pos_mean":  # bound the negative-patch sum on large grids
            hl = hl / (tgt["valid"][hm].sum(-1).float() / 100).clamp(min=1)
        L["heat"] = hl.mean()
        with torch.no_grad():  # selection diagnostics: heat probability on negatives / at positives, and its scale
            p = mo["heat"][hm].sigmoid()
            neg = tgt["valid"][hm] & ~tgt["pos"][hm]
            L["_heat_p_neg"] = (p * neg).sum() / neg.sum().clamp(min=1)
            L["_heat_p_pos"] = (p * tgt["pos"][hm]).sum() / tgt["pos"][hm].sum().clamp(min=1)
    else:
        L["heat"] = z
    cm = present & tgt["n_known"]
    if cm.any():
        cls = (tgt["n"][cm].clamp(1, cfg.count_classes) - 1).long()
        L["count"] = F.cross_entropy(mo["count_logits"][cm], cls)
        L["_count_acc"] = (mo["count_logits"][cm].argmax(-1) == cls).float().mean().detach()
    else:
        L["count"] = z
    bm = present & tgt["gmask"].any(-1)
    if bm.any():
        lab, _ = greedy_match_labels(mo["cand_box"][bm], mo["cand_ok"][bm], tgt["gt"][bm], tgt["gmask"][bm])
        ok = mo["cand_ok"][bm]
        L["match"] = F.binary_cross_entropy_with_logits(mo["cand_match_logit"][bm][ok], lab[ok]) if ok.any() else z
        pred = (mo["cand_match_logit"][bm].sigmoid() >= 0.5) & ok
        tp = (pred & (lab > 0)).sum(-1).float()
        f1 = 2 * tp / (pred.sum(-1) + tgt["gmask"][bm].sum(-1)).clamp(min=1)
        L["_set_f1"] = f1.mean().detach()
    else:
        L["match"] = z
    total = cfg.w_heat * L["heat"] + cfg.w_count * L["count"] + cfg.w_match * L["match"]
    return total, L


def nms_keep(boxes, scores, thr):
    """Greedy NMS over a short list. boxes [K,4], scores [K] -> kept indices (score order)."""
    order = scores.argsort(descending=True).tolist()
    keep = []
    for i in order:
        if all(float(box_iou_giou(boxes[i], boxes[j])[0]) < thr for j in keep):
            keep.append(i)
    return keep


def decode_sets(p_present, single_box, single_kappa, cand_box, cand_ok, match_p, thr=0.5, nms=0.7, count_p=None):
    """Per question (CPU tensors): p_present [M], single_box [M,4], single_kappa [M], cand_box [M,K,4], cand_ok [M,K],
    match_p [M,K], optional count_p [M,C] (P(n=1..C | present), switches to count-guided decoding).
    -> list of (boxes [n,4], conf [n]); same rules as eval.decode_row (mode "thr" / "count")."""
    out = []
    for m in range(len(p_present)):
        g = float(p_present[m])
        if g < 0.5:
            out.append((single_box.new_zeros(0, 4), single_box.new_zeros(0)))
            continue
        ok = cand_ok[m].nonzero()[:, 0]
        if count_p is not None and len(ok):
            o = ok[match_p[m, ok].argsort(descending=True, stable=True)]
            o = o[nms_keep_ordered(cand_box[m, o], nms)]
            n = int(count_p[m].argmax()) + 1
            if n >= count_p.shape[1]:
                sel = o[match_p[m, o] >= thr]
                o = sel if len(sel) >= n else o[:n]
            else:
                o = o[:n]
            out.append((cand_box[m, o], g * match_p[m, o]))
            continue
        sel = (cand_ok[m] & (match_p[m] >= thr)).nonzero()[:, 0]
        if len(sel):
            k = nms_keep(cand_box[m, sel], match_p[m, sel], nms)
            sel = sel[k]
            out.append((cand_box[m, sel], g * match_p[m, sel]))
        else:
            out.append((single_box[m][None], single_box.new_tensor([g * float(single_kappa[m])])))
    return out


def nms_keep_ordered(boxes, thr):
    """greedy NMS over boxes already sorted by score -> kept positions"""
    keep = []
    for i in range(len(boxes)):
        if all(float(box_iou_giou(boxes[i], boxes[j])[0]) < thr for j in keep):
            keep.append(i)
    return keep
