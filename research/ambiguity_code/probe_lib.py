"""Glance probe helpers: run m.ask and also capture the patch-center distribution per box question."""
import math, time
import torch
from glance import Box

def load():
    from glance import Glance
    return Glance.from_pretrained("Vision-Decision/Glance-4B-preview")

def ask_with_centers(m, img, qs):
    cap = {}
    orig = m._forward
    def fw(batch):
        out = orig(batch); cap['batch'] = batch; cap['out'] = out; return out
    m._forward = fw
    t = time.time()
    try:
        res = m.ask(img, qs)
    finally:
        m._forward = orig
    dt = time.time() - t
    b, out = cap['batch'], cap['out']
    valid = b['vis_valid'][0].cpu()
    centers = b['centers'][0][valid].cpu()          # [N,2] normalized (x,y)
    gh_gw = None
    extra = []
    for k, (q, r) in enumerate(zip(qs, res)):
        if not (isinstance(q, Box) or isinstance(q, str)):
            extra.append(None); continue
        cl = out['center_logits'][k][valid.to(out['center_logits'].device) if False else valid]
        p = torch.softmax(cl.float(), -1)
        extra.append(p)
    return res, centers, extra, dt

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    I = ix * iy; U = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - I
    return I / U if U > 0 else 0.0

def center_stats(p, centers, W, H, gt_boxes=None, topk=8, pred_box=None):
    """p [N] softmax over patches; centers [N,2] normalized. gt_boxes list of xyxy pixels."""
    N = p.numel()
    gw = len(torch.unique(centers[:, 0])); gh = N // gw
    G = p.view(gh, gw)
    mx = torch.nn.functional.max_pool2d(G[None, None], 5, 1, 2)[0, 0]
    peaks = (G >= mx) & (G >= 0.25 * G.max())
    pk = sorted([float(v) for v in G[peaks]], reverse=True)
    gt_free = dict(n_peaks=int(peaks.sum()), peak2_over_peak1=round(pk[1] / pk[0], 4) if len(pk) > 1 else 0.0)
    if pred_box is not None:
        px = centers[:, 0] * W; py = centers[:, 1] * H; b = pred_box
        inb = (px >= b[0]) & (px <= b[2]) & (py >= b[1]) & (py <= b[3])
        gt_free['mass_out_pred'] = round(1 - float(p[inb].sum()), 4)
        # mass outside the predicted box but within the box-sized neighbourhood is excluded: use a 1.5x enlarged box
        cxb, cyb, hw, hh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * 0.75, (b[3] - b[1]) * 0.75
        inb2 = (px >= cxb - hw) & (px <= cxb + hw) & (py >= cyb - hh) & (py <= cyb + hh)
        gt_free['mass_out_pred15'] = round(1 - float(p[inb2].sum()), 4)
    ent = float(-(p * (p.clamp_min(1e-12)).log()).sum())
    s = dict(n_patches=N, entropy=round(ent, 4), entropy_norm=round(ent / math.log(N), 4),
             perplexity=round(math.exp(ent), 2), top1=round(float(p.max()), 4), **gt_free)
    v, i = p.topk(min(topk, N))
    s['top'] = [[round(float(centers[j, 0]) * W, 1), round(float(centers[j, 1]) * H, 1), round(float(x), 4)] for x, j in zip(v, i)]
    if gt_boxes:
        px = centers[:, 0] * W; py = centers[:, 1] * H
        masses = []
        for bx in gt_boxes:
            inside = (px >= bx[0]) & (px <= bx[2]) & (py >= bx[1]) & (py <= bx[3])
            masses.append(round(float(p[inside].sum()), 4))
        s['inst_mass'] = masses
        # exclusive assignment: each patch to the smallest containing box, to avoid double-counting overlaps
        s['mass_in_any'] = round(float(sum(p[((px >= b[0]) & (px <= b[2]) & (py >= b[1]) & (py <= b[3]))].sum() for b in gt_boxes[:1])) if len(gt_boxes)==1 else None, 4) if len(gt_boxes)==1 else None
        anyin = torch.zeros(N, dtype=torch.bool)
        for bx in gt_boxes:
            anyin |= (px >= bx[0]) & (px <= bx[2]) & (py >= bx[1]) & (py <= bx[3])
        s['mass_in_any'] = round(float(p[anyin].sum()), 4)
        sm = sorted(masses, reverse=True)
        s['n_inst_mass_ge_0.1'] = sum(x >= 0.1 for x in masses)
        s['split_ratio'] = round(sm[1] / sm[0], 4) if len(sm) > 1 and sm[0] > 0 else 0.0
    return s
