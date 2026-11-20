"""DGQP-style statistics of the box output distributions (for a cross-fitted *learned* readout, the report's option (b)).
python arch/quality/stats.py IN.npz OUT.npz   (IN = arch/quality/dump.py output with top_* arrays)
Per item: top-4 probabilities + entropy + 10-90% quantile width of each of the 4 edge distributions at the argmax patch,
top-8 centre probabilities, and the centre-weighted agreement sum_j p_j IoU(box_j, box_hat) over the top-8 patches'
decoded point boxes (plus the max IoU of a non-argmax candidate with mass > 0.05)."""
import sys
import numpy as np

d = np.load(sys.argv[1])
el = d["top_el"].astype(np.float32)          # [M,8,4,K]
tp = d["top_p"].astype(np.float32)           # [M,8]
tc = d["top_c"].astype(np.float32)           # [M,8,2]
M, k, _, K = el.shape
p = np.exp(el - el.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
bins = np.linspace(-0.1, 1.0, K)
p0 = p[:, 0]                                  # [M,4,K]
top4 = -np.sort(-p0, -1)[..., :4]             # [M,4,4]
H = -(p0 * np.log(p0 + 1e-12)).sum(-1)        # [M,4]
cdf = p0.cumsum(-1)
q10 = bins[np.clip((cdf < 0.1).sum(-1), 0, K - 1)]; q90 = bins[np.clip((cdf < 0.9).sum(-1), 0, K - 1)]
width = q90 - q10                              # [M,4]
e = (p * bins).sum(-1)                        # [M,8,4] expected offsets
bx = np.stack([tc[..., 0] - e[..., 0], tc[..., 1] - e[..., 1], tc[..., 0] + e[..., 2], tc[..., 1] + e[..., 3]], -1)  # [M,8,4]


def iou(a, b):
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda t: np.clip(t[..., 2] - t[..., 0], 0, None) * np.clip(t[..., 3] - t[..., 1], 0, None)
    return inter / np.maximum(ar(a) + ar(b) - inter, 1e-9)


agree = iou(bx[:, :1], bx)                     # [M,8]
w = tp / tp.sum(-1, keepdims=True)
agree_w = (w * agree).sum(-1)
alt = np.where(tp[:, 1:] > 0.05, 1 - agree[:, 1:], 0).max(-1)   # strongest disagreeing candidate
F = np.concatenate([top4.reshape(M, -1), H, width, tp, agree_w[:, None], alt[:, None],
                    (bx[:, 0, 2] - bx[:, 0, 0])[:, None], (bx[:, 0, 3] - bx[:, 0, 1])[:, None]], 1)
np.savez_compressed(sys.argv[2], F=F.astype(np.float32), agree_w=agree_w.astype(np.float32), alt=alt.astype(np.float32))
print("stats", F.shape)
