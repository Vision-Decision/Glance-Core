"""AR IoU-head experiment, step 2 (VM; GPU optional): train a box-quality head on top of the frozen AR-digits model.

python -m ablation.ar_iou_head --feat_dir DIR --out DIR/head_scores.npz

Head = same family / width / loss as the gate's IoU head (ablation/dvcore/heads.py BoxHeads.iou: LayerNorm'd inputs,
Linear(in, 512) -> GELU -> Linear(512, 1), BCE on 1[IoU(decoded box, gt) >= 0.5]). Inputs (feats from
ablation/ar_iou_feats.py):
  full    LN(h_q) | LN(h_ans) | LN(h_end) | standardized scalars | fourier(box centre) (64, as the gate head)
  scalar  standardized scalars | fourier(box centre)              (no hidden states: what token probabilities buy)
scalars = answer length, box log-prob (sum, per token), min token log-prob, mean / min top-1 margin, first-token log q,
log q(none), first-token margin to none, box x1 y1 x2 y2 w h log-area.
Training sets: H = arhead_rcg (held-out RefCOCOg images), T = arhead_train (in-sample v0 training images).
Early stopping on a 10% image-held-out slice of the training set (best validation BCE over 40 epochs).
Outputs logits for grefcoco_val_merged and arcal_gtest for each variant (head_scores.npz, keys <variant>__<set>)
plus a small JSON with training curves.
"""
import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ablation.dvcore.heads import fourier

VARIANTS = {"heldout_full": ("arhead_rcg", True), "heldout_scalar": ("arhead_rcg", False),
            "insample_full": ("arhead_train", True)}
SETS = ["grefcoco_val_merged", "arcal_gtest"]


def load(d, name):
    rows = json.load(open(f"{d}/rows_{name}.json"))
    f = dict(np.load(f"{d}/feats_{name}.npz"))
    L = f["L"]
    tl, tm = f["tok_lp"], f["tok_mg"]
    mask = np.arange(tl.shape[1])[None] < L[:, None]
    box = np.array([r["box"] if r["box"] else [0, 0, 0, 0] for r in rows], np.float32)
    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    pmc = np.array([r.get("p_maxcoord_c", r["p_maxcoord"]) for r in rows])
    pn = np.array([r["p_none"] for r in rows])
    lpb = tl.sum(1)
    sc = np.stack([L, lpb, lpb / L, np.where(mask, tl, 0).min(1), (tm * mask).sum(1) / L,
                   np.where(mask, tm, 1e9).min(1), np.log(np.clip(pmc, 1e-30, 1)), np.log(np.clip(pn, 1e-30, 1)),
                   np.log(np.clip(pmc, 1e-30, 1)) - np.log(np.clip(pn, 1e-30, 1)),
                   box[:, 0], box[:, 1], box[:, 2], box[:, 3], w, h, np.log(np.clip(w * h, 1e-6, 1))], 1).astype(np.float32)
    ctr = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1)
    lab = np.array([r["present"] and "iou" in r for r in rows])
    y = np.array([r.get("iou", 0.0) >= .5 for r in rows], np.float32)
    img = np.array([r["image_id"] for r in rows])
    return dict(h=np.concatenate([f["h_q"], f["h_ans"], f["h_end"]], 1), sc=sc, ctr=ctr.astype(np.float32),
                lab=lab, y=y, img=img, D=f["h_q"].shape[1])


class Head(nn.Module):
    def __init__(self, D, n_sc, full, d=512):
        super().__init__()
        self.full, self.D = full, D
        self.ln = nn.ModuleList([nn.LayerNorm(D) for _ in range(3)]) if full else None
        self.sc_ln = nn.LayerNorm(n_sc)
        self.mlp = nn.Sequential(nn.Linear((3 * D if full else 0) + n_sc + 64, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, sc, ctr):
        x = [self.sc_ln(sc), fourier(ctr)]
        if self.full:
            x = [ln(h[:, k * self.D:(k + 1) * self.D]) for k, ln in enumerate(self.ln)] + x
        return self.mlp(torch.cat(x, -1)).squeeze(-1)


def tens(d, idx, dev):
    return (torch.tensor(d["h"][idx], dtype=torch.float32, device=dev), torch.tensor(d["sc"][idx], device=dev),
            torch.tensor(d["ctr"][idx], device=dev), torch.tensor(d["y"][idx], device=dev))


@torch.no_grad()
def predict(net, d, dev, bs=4096):
    out = []
    for i in range(0, len(d["y"]), bs):
        idx = np.arange(i, min(i + bs, len(d["y"])))
        h, sc, c, _ = tens(d, idx, dev)
        out.append(net(h, sc, c).float().cpu().numpy())
    return np.concatenate(out)


def auroc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train(d, full, dev, seed=0, epochs=40, bs=256, lr=3e-4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.where(d["lab"])[0]
    imgs = np.unique(d["img"][idx])
    va_img = set(rng.choice(imgs, max(1, len(imgs) // 10), replace=False).tolist())
    va = np.array([i for i in idx if d["img"][i] in va_img]); tr = np.array([i for i in idx if d["img"][i] not in va_img])
    net = Head(d["D"], d["sc"].shape[1], full).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    Hv, Sv, Cv, Yv = tens(d, va, dev)
    best, curve = None, []
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            h, sc, c, y = tens(d, perm[i:i + bs], dev)
            loss = F.binary_cross_entropy_with_logits(net(h, sc, c), y)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            lv = net(Hv, Sv, Cv)
            vl = float(F.binary_cross_entropy_with_logits(lv, Yv))
        curve.append(dict(epoch=ep, val_bce=vl, val_auroc=auroc(lv.cpu().numpy(), Yv.cpu().numpy())))
        if best is None or vl < best[0]:
            best = (vl, ep, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[2]); net.eval()
    return net, dict(n_train=int(len(tr)), n_val=int(len(va)), pos_rate=float(d["y"][idx].mean()), best_epoch=best[1],
                     best_val_bce=best[0], best_val_auroc=curve[best[1]]["val_auroc"], curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = {n: load(a.feat_dir, n) for n in set(SETS) | {v[0] for v in VARIANTS.values()}}
    res, info = {}, {}
    for name, (src, full) in VARIANTS.items():
        net, inf = train(data[src], full, dev)
        info[name] = inf
        for s in SETS:
            res[f"{name}__{s}"] = predict(net, data[s], dev)
            d = data[s]
            m = d["lab"]
            inf[f"auroc_{s}"] = auroc(res[f"{name}__{s}"][m], d["y"][m])
        print(name, {k: v for k, v in inf.items() if k != "curve"}, flush=True)
    for s in SETS:
        res[f"img__{s}"] = data[s]["img"]
    np.savez(a.out, **res)
    json.dump(info, open(a.out.replace(".npz", "_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
