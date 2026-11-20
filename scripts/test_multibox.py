"""CPU unit tests for the v1.2 multi-instance box readout (dv/model/multibox.py) on tiny random tensors; no model weights.

python scripts/test_multibox.py

Checks: peak finder vs brute force; heat target; greedy set labels (duplicates -> 0); padded GT; the multi-referent
centre target equals the old single-box target when N = 1; old checkpoints load and the single-box outputs are unchanged
by adding the multi heads; in train_only=multi only multi.* tensors get gradients; collate carries all referent boxes;
set metrics on hand-made rows; and a small synthetic task where the heads must learn to return every planted instance.
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from dv.config import Config
from dv.model.heads import BoxHeads, center_target_box, box_loss_v1
from dv.model.multibox import (local_peaks, heat_target, greedy_match_labels, multi_loss, focal_heat_loss)
from dv.model.v1forward import pad_gt, multi_center_target

torch.manual_seed(0)
ok = lambda c, msg: print(("PASS " if c else "FAIL ") + msg) or (c or sys.exit(1))


def grid(gh, gw, N):
    ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    c = torch.zeros(N, 2); v = torch.zeros(N, dtype=torch.bool)
    c[:gh * gw, 0] = (jj.flatten() + 0.5) / gw; c[:gh * gw, 1] = (ii.flatten() + 0.5) / gh; v[:gh * gw] = True
    return c, v


# ---------------------------------------------------------------- 1. local peaks == brute force
N = 40
shapes = [(5, 7), (6, 6), (4, 9), (1, 5)]
cen, val = zip(*[grid(h, w, N) for h, w in shapes])
cen, val = torch.stack(cen), torch.stack(val)
ghw = torch.tensor(shapes)
s = torch.randn(len(shapes), N).round(decimals=1)  # rounding creates ties
s = s.masked_fill(~val, -1e4)
idx, okk = local_peaks(s, ghw, val, 50)
good = True
for m, (gh, gw) in enumerate(shapes):
    want = set()
    for r in range(gh):
        for c in range(gw):
            j = r * gw + c
            peak = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr or dc) and 0 <= rr < gh and 0 <= cc < gw:
                        nj = rr * gw + cc
                        if s[m, nj] > s[m, j] or (s[m, nj] == s[m, j] and nj < j):
                            peak = False
            if peak:
                want.add(j)
    got = set(idx[m][okk[m]].tolist())
    good &= got == want
    vals = s[m, idx[m][okk[m]]]
    good &= bool((vals[:-1] >= vals[1:]).all())
ok(good, "local_peaks matches brute-force 8-neighbourhood maxima (with ties), sorted by logit")

# ---------------------------------------------------------------- 2. heat target
gt = torch.tensor([[[0.0, 0.0, 0.3, 0.3], [0.6, 0.6, 1.0, 1.0]], [[0.2, 0.2, 0.5, 0.9], [0, 0, 0, 0]]])
gm = torch.tensor([[True, True], [True, False]])
c2, v2 = zip(*[grid(6, 6, 36), grid(6, 6, 36)])
c2, v2 = torch.stack(c2), torch.stack(v2)
g2 = torch.tensor([[6, 6], [6, 6]])
Y, pos = heat_target(gt, gm, c2, v2, g2)
ok(pos.sum(-1).tolist() == [2, 1] and bool((Y[pos] == 1).all()) and float(Y.max()) <= 1 and float(Y.min()) >= 0,
   "heat target: one positive per referent centre, Y in [0,1]")
ok(int(pos[0].nonzero()[0]) == 0 and int(pos[0].nonzero()[1]) == 4 * 6 + 4, "heat target: positives at the centre patches")
l = focal_heat_loss(torch.where(pos, 8.0, -8.0), Y, pos, v2)
l2 = focal_heat_loss(torch.zeros(2, 36), Y, pos, v2)
ok(bool((l < l2).all()), "focal heat loss lower for a correct heat map")

# ---------------------------------------------------------------- 3. greedy set labels
cb = torch.tensor([[[0.6, 0.6, 1.0, 1.0], [0.62, 0.6, 1.0, 1.0], [0.0, 0.0, 0.3, 0.32], [0.4, 0.0, 0.5, 0.1]]])
lab, best = greedy_match_labels(cb, torch.ones(1, 4, dtype=torch.bool), gt[:1], gm[:1])
ok(lab.tolist() == [[1.0, 0.0, 1.0, 0.0]], f"set labels: duplicate of a matched referent gets 0 ({lab.tolist()})")

# ---------------------------------------------------------------- 4. padded GT + multi-referent centre target
gb_m = torch.tensor([3, 1, 3, 3])
gb = torch.rand(4, 4).sort(-1).values[:, [0, 1, 2, 3]]
bi = torch.tensor([1, 3, 4])
P, PM = pad_gt(gb_m, gb, bi, 5)
ok(P.shape == (3, 3, 4) and PM.sum(1).tolist() == [1, 3, 0] and torch.equal(P[1, :3], gb[[0, 2, 3]]) and torch.equal(P[0, 0], gb[1]),
   "pad_gt groups referent boxes by question, keeps order, masks padding")
b1 = torch.tensor([[0.1, 0.2, 0.5, 0.7], [0.3, 0.3, 0.9, 0.6]])
ct_old = center_target_box(b1, c2, v2, g2)
ct_new, cov = multi_center_target(b1[:, None], torch.ones(2, 1, dtype=torch.bool), c2, v2, g2)
ok(torch.allclose(ct_old, ct_new), "single referent: centre target identical to the pre-v1.2 target")
ct3, _ = multi_center_target(gt, gm, c2, v2, g2)
ok(torch.allclose(ct3.sum(-1), torch.ones(2)) and abs(float(ct3[0, :18].sum()) - 0.5) < 0.2,
   "two referents: centre mass split between them, sums to 1")

# ---------------------------------------------------------------- 5. backward compatibility of the heads
D = 32
cfg0 = Config().update(dict(head_dim=16, edge_bins=8))
cfg1 = cfg0.update(dict(multibox=True, multi_k=6, multi_k_train=4))
old = BoxHeads(D, cfg0); new = BoxHeads(D, cfg1)
res = new.load_state_dict(old.state_dict(), strict=False)
ok(all(".multi." in "." + k for k in res.missing_keys) and not res.unexpected_keys and res.missing_keys,
   "pre-v1.2 head state dict loads; only multi.* missing")
new.multi.init_from(new)
B, Nv = 2, 36
vis = torch.randn(B, Nv, D); vv = v2.clone(); cc = c2.clone()
M = 5
hq = torch.randn(M, D); q_img = torch.tensor([0, 0, 1, 1, 1])
ghw = g2[q_img]
o0 = old(hq, vis, vv, cc, q_img)
o1 = new(hq, vis, vv, cc, q_img, grid_hw=ghw, multi_k=6)
same = all(torch.equal(o0[k], o1[k]) for k in ("presence_logit", "center_logits", "box_pred", "iou_logit", "pred_idx"))
ok(same, "adding the multi heads leaves presence / centre / box / IoU outputs bit-identical")
mo = o1["multi"]
ok(bool((mo["heat"].argmax(-1) == o1["pred_idx"]).all()),
   "heat map initialised from the centre softmax: same arg-max patch")
ok(mo["cand_box"].shape == (M, 6, 4) and bool(mo["cand_ok"][:, 0].all()), "candidates: shape and first peak valid")
ok(torch.allclose(mo["cand_box"][torch.arange(M), 0], o1["box_pred"], atol=1e-5),
   "top peak candidate box == single-box answer at init")

# gradients only into multi.* in train_only=multi; the box-loss path is detached as in train.py
for p in new.parameters():
    p.grad = None
gtq, gmq = pad_gt(torch.tensor([0, 0, 2, 3]), torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9],
                                                           [0.2, 0.2, 0.6, 0.6], [0.0, 0.3, 0.3, 0.9]]), torch.arange(M), M)
Yq, posq = heat_target(gtq, gmq, cc[q_img], vv[q_img], ghw)
tgt = dict(present=gmq.any(1), gt=gtq, gmask=gmq, Y=Yq, pos=posq, valid=vv[q_img], n=gmq.sum(1), n_known=gmq.any(1))
cfgT = cfg1.update(dict(train_only="multi"))
ml, parts = multi_loss(mo, tgt, cfgT)
ml.backward()
with_grad = sorted({n.split(".")[0] + "." + n.split(".")[1] if n.startswith("multi") else n.split(".")[0]
                    for n, p in new.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0})
ok(all(n.startswith("multi") or n in ("q_norm", "v_norm") for n in with_grad) and "multi.h_q" in with_grad
   and "multi.match" in with_grad and "multi.count" in with_grad,
   f"multi loss reaches multi.* and the shared input norms only, never presence / centre / edge / IoU ({with_grad})")

# box_loss_v1 with several referents: kappa target is hit-any
sel = torch.zeros(M, dtype=torch.long)
o2 = new(hq, vis, vv, cc, q_img, sel_idx=sel)
bt = dict(present=tgt["present"], has_box=tgt["present"], box=gtq[:, 0], center_t=multi_center_target(gtq, gmq, cc[q_img], vv[q_img], ghw)[0],
          gt=gtq, gmask=gmq)
tot, L = box_loss_v1(o2, bt, cfg1)
ok(torch.isfinite(tot).item(), "box_loss_v1 with padded referents is finite")

# ---------------------------------------------------------------- 6. collate carries every referent box
from dv.data.mixed import collate_mixed
tok = types.SimpleNamespace(pad=0, pad_img=1, pad_vid=2)
def item(qboxes):
    return dict(segs=[("ids", [5, 6]), ("img", None)], grid_thw=torch.tensor([1, 8, 8]), pixel_values=torch.zeros(64, 4),
                q_ids=[[7, 8]] * len(qboxes), qtype=[0] * len(qboxes), labs=[[]] * len(qboxes), tgts=[[]] * len(qboxes),
                present=[bool(b) for b in qboxes], has_box=[bool(b) for b in qboxes],
                box=[b[0] if b else [0, 0, 0, 0] for b in qboxes], points=[[]] * len(qboxes), gt_boxes=qboxes,
                n_tgt=[len(b) for b in qboxes], qs=[{}] * len(qboxes), orig_wh=(10, 10), rec={})
bx = lambda i: [0.1 * i, 0.1 * i, 0.1 * i + 0.1, 0.1 * i + 0.1]
b = collate_mixed([item([[bx(0)], [bx(1), bx(2), bx(3)]]), item([[], [bx(4), bx(5)]])], tok)
ok(b["gb_m"].tolist() == [0, 1, 1, 1, 3, 3] and b["n_tgt"].tolist() == [1, 3, 0, 2] and torch.allclose(b["gb_xyxy"][4], torch.tensor(bx(4))),
   "collate: flat referent boxes with question index, n_tgt")
legacy = item([[bx(0)], []]); del legacy["gt_boxes"]; del legacy["n_tgt"]
b = collate_mixed([legacy], tok)
ok(b["gb_m"].tolist() == [0] and b["n_tgt"].tolist() == [1, 0], "collate: items without gt_boxes fall back to the single box")

# ---------------------------------------------------------------- 7. set metrics
# eval.py imports the model stack (transformers / peft); load only its metric functions
import numpy as np, collections
code = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval.py")).read()
ns = dict(np=np, collections=collections)
exec(code[code.index("def ece("):code.index("@torch.no_grad()")], ns)
exec(code[code.index("def _iou("):code.index("def summarize_grec")], ns)
G3 = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
rows = [
    dict(kind="box", present=True, p_present=0.9, p_iou50=0.8, box=[0, 0, 10, 10], gt_boxes=G3, n_targets=3),       # single box (v1.1-like)
    dict(kind="box", present=True, p_present=0.9, p_iou50=0.8, box=[0, 0, 10, 10], gt_boxes=G3, n_targets=3,
         cands=[g + [3.0, 0.9, 0.9] for g in G3] + [[1, 0, 10, 10, 1.0, 0.1, 0.9]], count_p=[0.05, 0.05, 0.8, 0.1]),
    dict(kind="box", present=False, p_present=0.1, p_iou50=0.8, box=[0, 0, 10, 10], gt_boxes=[], n_targets=0),
    dict(kind="box", present=True, p_present=0.9, p_iou50=0.8, box=[0, 0, 10, 10], gt_box=[0, 0, 10, 11], n_targets=1),
]
sm = ns["summarize_multi"](rows[:1] + rows[2:])
ok(abs(sm["multi"]["f1"] - 0.5) < 1e-9 and sm["N_acc"] == 1.0 and sm["single"]["f1"] == 1.0 and sm["multi"]["count_acc"] == 0,
   "set metrics: single box on 3 referents -> F1 0.5; empty on no-target -> 1; exact single -> 1")
sm = ns["summarize_multi"](rows[1:])
ok(sm["multi"]["f1"] == 1.0 and sm["multi"]["count_acc"] == 1.0 and sm["count_head"]["acc"] == 1.0,
   "set metrics: decoded candidates (dup suppressed by P(match)) -> F1 1, count right")

from dv.model.multibox import decode_sets
torch.manual_seed(3)
same = True
for _ in range(50):
    K = 6
    cb = torch.rand(1, K, 4).sort(-1).values[:, :, [0, 1, 2, 3]] * 100
    cb = torch.cat([torch.minimum(cb[..., :2], cb[..., 2:]), torch.maximum(cb[..., :2], cb[..., 2:]) + 1], -1)
    mp = torch.rand(1, K); okc = torch.rand(1, K) > 0.2; g = float(torch.rand(1)) * 0.6 + 0.4
    sb = torch.tensor([[1.0, 2.0, 30.0, 40.0]])
    (bs, cf), = decode_sets(torch.tensor([g]), sb, torch.tensor([0.7]), cb, okc, mp, 0.5)
    row = dict(p_present=g, p_iou50=0.7, box=sb[0].tolist(),
               cands=[cb[0, k].tolist() + [0.0, float(mp[0, k]), 0.5] for k in range(K) if okc[0, k]])
    rb, rc = ns["decode_row"](row, 0.5)
    same &= len(rb) == len(bs) and all(abs(x - y) < 1e-3 for b1, b2 in zip(rb, bs.tolist()) for x, y in zip(b1, b2))
ok(same, "release decode_sets == eval decode_row on random candidates")
same = True
for _ in range(50):
    K = 6
    cb = torch.rand(1, K, 4) * 50; cb = torch.cat([cb[..., :2], cb[..., :2] + 5 + cb[..., 2:]], -1)
    mp = torch.rand(1, K); okc = torch.rand(1, K) > 0.2; cp = torch.rand(1, 4).softmax(-1); g = 0.9
    sb = torch.tensor([[1.0, 2.0, 30.0, 40.0]])
    (bs, cf), = decode_sets(torch.tensor([g]), sb, torch.tensor([0.7]), cb, okc, mp, 0.3, count_p=cp)
    row = dict(p_present=g, p_iou50=0.7, box=sb[0].tolist(), count_p=cp[0].tolist(),
               cands=[cb[0, k].tolist() + [0.0, float(mp[0, k]), 0.5] for k in range(K) if okc[0, k]])
    rb, rc = ns["decode_row"](row, 0.3, mode="count")
    same &= len(rb) == len(bs) and all(abs(x - y) < 1e-3 for b1, b2 in zip(rb, bs.tolist()) for x, y in zip(b1, b2))
ok(same, "count-guided decoding: release decode_sets == eval decode_row")

# ---------------------------------------------------------------- 8. synthetic learning check (heads only)
# Visual tokens carry a "category" vector at each planted instance's patches; the question vector names one category.
# The multi heads (trained alone, everything else frozen at random init) must learn to return every instance.
torch.manual_seed(1)
D, gh, gw = 32, 8, 8
Nv = gh * gw
cfgL = Config().update(dict(head_dim=32, edge_bins=16, multibox=True, multi_k=8, multi_k_train=8, train_only="multi",
                             lr_heads=3e-3))
H = BoxHeads(D, cfgL); H.multi.init_from(H)
cat = torch.randn(4, D) * 2
cG, vG = grid(gh, gw, Nv)


def sample(nimg=16):
    vis = torch.randn(nimg, Nv, D) * 0.5
    hq, qi, boxes = [], [], []
    for b in range(nimg):
        c = int(torch.randint(4, (1,)))
        n = int(torch.randint(0, 4, (1,)))
        cells = torch.randperm(16)[:n]
        bl = []
        for cell in cells.tolist():
            r0, c0 = (cell // 4) * 2, (cell % 4) * 2
            for dr in (0, 1):
                for dc in (0, 1):
                    vis[b, (r0 + dr) * gw + c0 + dc] += cat[c]
            bl.append([c0 / gw, r0 / gh, (c0 + 2) / gw, (r0 + 2) / gh])
        hq.append(cat[c] + 0.1 * torch.randn(D)); qi.append(b); boxes.append(bl)
    return vis, torch.stack(hq), torch.tensor(qi), boxes


def train_batch(vis, hq, qi, boxes):
    Mq = len(qi)
    gbm = torch.tensor([m for m, bl in enumerate(boxes) for _ in bl], dtype=torch.long)
    gbx = torch.tensor([b for bl in boxes for b in bl], dtype=torch.float).reshape(-1, 4)
    gtq, gmq = pad_gt(gbm, gbx, torch.arange(Mq), Mq)
    vq, cq = vG[None].expand(Mq, -1), cG[None].expand(Mq, -1, -1)
    gq = torch.tensor([[gh, gw]] * Mq)
    Yq, posq = heat_target(gtq, gmq, cq, vq, gq)
    return gtq, gmq, dict(present=gmq.any(1), gt=gtq, gmask=gmq, Y=Yq, pos=posq, valid=vq, n=gmq.sum(1), n_known=gmq.any(1)), gq


opt = torch.optim.Adam([p for p in H.multi.parameters()], lr=3e-3)
for p in H.parameters():
    if not any(p is q for q in H.multi.parameters()):
        p.requires_grad_(False)
# the edge head is frozen at random init in this toy, so supervise the match head with the *true* geometry instead:
# replace candidate boxes by the box of the planted instance under each peak (tests heat/peaks/match/count, not edges)
def oracle_boxes(mo, boxes):
    out = mo["cand_box"].clone()
    for m, bl in enumerate(boxes):
        for k, j in enumerate(mo["cand_idx"][m].tolist()):
            r, c = j // gw, j % gw
            out[m, k] = torch.tensor([(c // 2) * 2 / gw, (r // 2) * 2 / gh, ((c // 2) * 2 + 2) / gw, ((r // 2) * 2 + 2) / gh])
    return out
f1s = []
for it in range(400):
    vis, hq, qi, boxes = sample()
    gtq, gmq, tgt, gq = train_batch(vis, hq, qi, boxes)
    o = H(hq, vis, vG[None].expand(len(qi), -1)[:16], cG[None].expand(16, -1, -1), qi, grid_hw=gq, multi_k=8)
    o["multi"]["cand_box"] = oracle_boxes(o["multi"], boxes)
    l, parts = multi_loss(o["multi"], tgt, cfgL)
    opt.zero_grad(); l.backward(); opt.step()
    if "_set_f1" in parts:
        f1s.append(float(parts["_set_f1"]))
early, late = sum(f1s[:20]) / 20, sum(f1s[-50:]) / 50
cacc = float(parts.get("_count_acc", 0))
print(f"     synthetic: set F1 (match>=0.5) first 20 steps {early:.3f} -> last 50 steps {late:.3f}; count acc {cacc:.2f}")
ok(late > 0.9 and late > early + 0.2, "synthetic multi-instance task: heads learn to return every instance")
print("ALL PASS")
