"""GDN diagnostics on Glance (zero training), with the prefix-cache engine arch/zoom/lite.py:

  modes   full        normal readout
          attn_noimg  the question block may NOT attend to the image tokens in the 8 full-attention layers; image
                      information can reach the question only through the GDN recurrent state (+ conv tail)
          gdn_reset   the question block starts every GDN layer from a zero state; image information reaches it only
                      through the full-attention layers
  The box head always reads the (unchanged) visual hidden states; what changes is the question readout hq.
  python -m arch.gdn.ablate --ckpt CKPT --bench needle_64.jsonl[,..] [--coco data/prep/eval_refcoco_val.jsonl --limit 300]
         --max_vis 1600 --out DIR/name.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from PIL import Image

from dv.config import Config
from dv.model.build import build, load_ckpt
from dv.data.dataset import resize_dims
from arch.zoom.run_zoom import Zoom, inside
from arch.zoom.lite import grid_centers

MODES = ["full", "attn_noimg", "gdn_reset"]


def auroc(s, y):
    from scipy.stats import rankdata
    s, y = np.asarray(s), np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


@torch.no_grad()
def run_set(Z, recs, max_vis, up):
    out = {m: [] for m in MODES}
    for r in recs:
        img = Image.open(os.path.join(r["image_root"], r["file"])).convert("RGB")
        W, H = img.size
        w, h = resize_dims(W, H, max_vis, up)
        feats, grids = Z.vit([img.resize((w, h), Image.BICUBIC)])
        gh, gw = grids[0]
        cache, vis, q_start = Z.prefix(feats[0], gh, gw)
        Lp = next(c[1].shape[2] for c in cache if c[0] == "attn")
        keep = torch.ones(Lp, dtype=torch.bool)
        n_head = len(Z.tok.user_open) + 1
        keep[n_head:n_head + gh * gw] = False
        texts = [q["text"] for q in r["questions"]]
        for m in MODES:
            kw = dict(prefix_keep=keep) if m == "attn_noimg" else (dict(gdn_reset=True) if m == "gdn_reset" else {})
            hq = plain(Z, cache, q_start, texts, **kw)
            o = Z.head(hq, [vis], [grid_centers(gh, gw)])
            bx = (o["box_pred"].clamp(0, 1).float().cpu() * torch.tensor([W, H, W, H])).tolist()
            for j, q in enumerate(r["questions"]):
                g = q.get("box")
                out[m].append(dict(present=bool(q["present"]), p=float(o["presence_logit"][j]), N=q.get("meta", {}).get("N"),
                                   hit=inside(bx[j], g) if g else None, iou=iou(bx[j], g) if g else None))
    S = {}
    for m, rows in out.items():
        pr = [x for x in rows if x["present"] and x["hit"] is not None]
        S[m] = dict(n=len(rows), click_acc=float(np.mean([x["hit"] for x in pr])) if pr else None,
                    acc50=float(np.mean([x["iou"] >= 0.5 for x in pr])) if pr else None,
                    presence_auroc=auroc([x["p"] for x in rows], [x["present"] for x in rows]),
                    recall_0=float(np.mean([x["p"] > 0 for x in rows if x["present"]])) if any(x["present"] for x in rows) else None,
                    N_acc=float(np.mean([x["p"] < 0 for x in rows if not x["present"]])) if any(not x["present"] for x in rows) else None)
    return S


def plain(Z, cache, q_start, texts, **kw):
    bl = [Z.tok.box_block(x) for x in texts]
    T = max(map(len, bl)); M = len(bl)
    ids = torch.full((M, T), Z.tok.pad, dtype=torch.long); valid = torch.zeros(M, T, dtype=torch.bool)
    for i, b in enumerate(bl):
        ids[i, :len(b)] = torch.tensor(b); valid[i, :len(b)] = True
    pos = (q_start + torch.arange(T)).repeat(3, M, 1)
    within = valid[:, None, :].expand(M, T, T) | torch.eye(T, dtype=torch.bool)[None]
    h = Z.E.blocks(cache, ids.cuda(), pos.cuda(), within.cuda(), **kw)
    return h[torch.arange(M).cuda(), torch.tensor([len(b) - 1 for b in bl]).cuda()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", default="")
    ap.add_argument("--coco", default="")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max_vis", type=int, default=1600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__}).update(dict(grad_ckpt=False, max_q_tokens=96))
    net, heads, proc = build(cfg, "cuda"); load_ckpt(a.ckpt, net, heads); net.eval(); heads.eval()
    Z = Zoom(net, heads, proc, cfg)
    res = dict(ckpt=a.ckpt, max_vis=a.max_vis, sets={})
    for b in [x for x in a.bench.split(",") if x]:
        recs = [json.loads(l) for l in open(b)]
        res["sets"][os.path.basename(b)] = run_set(Z, recs, a.max_vis, 1.0)
        print("[ablate]", b, json.dumps(res["sets"][os.path.basename(b)]), flush=True)
    if a.coco:
        recs = []
        for l in list(open(a.coco))[: a.limit]:
            r = json.loads(l)
            recs.append(dict(file=r["file"], image_root="/content/data/coco/train2014", wh=[r["W"], r["H"]], questions=r["questions"]))
        res["sets"]["refcoco"] = run_set(Z, recs, 640, 2.0)
        print("[ablate] refcoco", json.dumps(res["sets"]["refcoco"]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
