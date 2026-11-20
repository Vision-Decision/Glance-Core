"""Box quality read off the box output itself (no learned quality head).

Glance decodes a box from (i) a softmax over patches for the centre patch and (ii) four ordinal distributions over edge
offsets (left, top, right, bottom) relative to the chosen patch centre, trained with a Cramer (CDF-L2) loss. Treat that
output as a predictive distribution over the true box B and score the point answer b_hat (expected box at the argmax
patch) by
    E[IoU(b_hat, B)]   and   P(IoU(b_hat, B) >= 0.5),
estimated by Monte Carlo (S samples; bins jittered uniformly within the bin width). Two predictive families:
    edge : B from the four edge distributions at the argmax patch only (localisation uncertainty)
    mix  : patch j ~ centre softmax restricted to the top-K patches (renormalised), then edges at patch j
           (localisation + "which object" uncertainty)
and mix_full = P50 under mix times the top-K centre mass (mass outside the top-K counted as misses).
Everything here is a deterministic function of the logits the model already produces: zero training, ~0 extra compute.
"""
import torch


def sample_boxes(edge_logits, pc, bin_centers, S, gen=None, temp=1.0):
    """edge_logits [M,4,K], pc [M,2] -> boxes [M,S,4] (xyxy, normalised)."""
    M, _, K = edge_logits.shape
    p = (edge_logits.float() / temp).softmax(-1).reshape(M * 4, K)
    idx = torch.multinomial(p, S, replacement=True, generator=gen).view(M, 4, S)
    bw = float(bin_centers[1] - bin_centers[0])
    off = bin_centers[idx] + (torch.rand(idx.shape, device=idx.device, generator=gen) - 0.5) * bw  # [M,4,S]
    x1 = pc[:, 0, None] - off[:, 0]; y1 = pc[:, 1, None] - off[:, 1]
    x2 = pc[:, 0, None] + off[:, 2]; y2 = pc[:, 1, None] + off[:, 3]
    return torch.stack([x1, y1, x2, y2], -1)


def iou_b(a, b):
    """a [...,4], b [...,4] broadcast."""
    x1 = torch.maximum(a[..., 0], b[..., 0]); y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2]); y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    ar = lambda t: (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)
    return inter / (ar(a) + ar(b) - inter).clamp(min=1e-9)


@torch.no_grad()
def quality(bh, hq, vis, centers, q_img, center_logits, box_pred, K=32, S=512, seed=0, temps=(1.0,)):
    """bh: BoxHeads; hq [M,D] raw readout; vis [B,N,D] raw visual hidden; centers [B,N,2]; center_logits [M,N];
    box_pred [M,4] (the model's answer). Returns dict of [M] tensors (+ top-8 raw arrays for offline analysis)."""
    gen = torch.Generator(device=hq.device).manual_seed(seed)
    M = hq.shape[0]
    hqn = bh.q_norm(hq.float()); visn = bh.v_norm(vis.float())
    p = center_logits.float().softmax(-1)
    k = min(K, p.shape[-1])
    tp, ti = p.topk(k, -1)                                                     # [M,k]
    qi = q_img[:, None].expand(-1, k)
    v_sel = visn[qi, ti]; pc = centers[qi, ti]                                 # [M,k,D], [M,k,2]
    el = bh.edges_logits(hqn[:, None].expand(-1, k, -1).reshape(M * k, -1), v_sel.reshape(M * k, -1),
                         pc.reshape(M * k, 2)).view(M, k, 4, -1)
    bc = bh.bin_centers.float()
    out = dict(top_mass=tp.sum(-1), top1=tp[:, 0])
    ent_e = -(el[:, 0].float().softmax(-1) * el[:, 0].float().log_softmax(-1)).sum(-1)   # [M,4]
    out["H_edges"] = ent_e.mean(-1)
    out["H_center"] = -(p * p.clamp(min=1e-30).log()).sum(-1)
    bhat = box_pred.float()
    for T in temps:
        sfx = "" if T == 1.0 else f"_T{T:g}"
        # edge-only at the argmax patch
        Bs = sample_boxes(el[:, 0], pc[:, 0], bc, S, gen, T)                  # [M,S,4]
        iou = iou_b(bhat[:, None], Bs)
        out["E_edge" + sfx] = iou.mean(-1); out["P50_edge" + sfx] = (iou >= 0.5).float().mean(-1)
        # mixture over the top-k patches
        w = tp / tp.sum(-1, keepdim=True)
        j = torch.multinomial(w, S, replacement=True, generator=gen)          # [M,S]
        Ball = sample_boxes(el.reshape(M * k, 4, -1), pc.reshape(M * k, 2), bc, S, gen, T).view(M, k, S, 4)
        Bm = Ball[torch.arange(M, device=j.device)[:, None], j, torch.arange(S, device=j.device)[None]]  # [M,S,4]
        iou = iou_b(bhat[:, None], Bm)
        out["E_mix" + sfx] = iou.mean(-1); out["P50_mix" + sfx] = (iou >= 0.5).float().mean(-1)
        out["P50_mixfull" + sfx] = out["P50_mix" + sfx] * out["top_mass"]
    k8 = min(8, k)
    raw = dict(top_p=tp[:, :k8].half(), top_i=ti[:, :k8].int(), top_c=pc[:, :k8].half(), top_el=el[:, :k8].half())
    return out, raw
