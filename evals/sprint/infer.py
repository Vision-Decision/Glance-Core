"""GUI grounding sprint: one inference pass of a v1/v1.1 checkpoint over a GUI benchmark, dumping rich per-question
outputs so that click-point decoders, multi-scale ensembles and crop-and-zoom can be scored offline.

python evals/sprint/infer.py --ckpt ckpt.pt --bench /content/gui/sspro.jsonl --max_vis 8192 --upscale 1.0 \
    --out /content/sprint/v1_sspro_8192            # -> .rows.jsonl.gz + .maps.npz

Zoom pass (second prefill on a crop around a first-pass point):
    --zoom_from <first-pass prefix> --zoom_frac 0.4 --zoom_point box|patch   (crop in full-image px; outputs mapped back)

Per question (rows): id, meta, wh, grid (gh,gw), crop, p_logit, p_iou, box (expected box at argmax patch, px),
top-k patches (prob, center px, decoded box px), center entropy. maps.npz: center log-probs (fp16) per row, so
scale ensembles can be formed by resampling onto a common pixel grid.
Model code is imported read-only from dv/ (same forward as eval.py).
"""
import argparse, gzip, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from PIL import Image

from dv.config import Config
from dv.data.mixed import MixedDataset, collate_mixed, TokenBudgetSampler
from dv.data.dataset import resize_dims
from dv.model.build import build, load_ckpt

Image.MAX_IMAGE_PIXELS = None


def to_dev(b, d):
    return {k: (v.to(d, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def load_bench(path, max_vis, upscale, limit=0):
    recs = []
    for l in open(path):
        r = json.loads(l)
        w, h = resize_dims(*r["wh"], max_vis, upscale)
        # plen must be the real prefix length: the token-budget sampler packs images by it (plen=0 packs max_records)
        recs.append(dict(src="gui_eval", img=os.path.join(r["image_root"], r["file"]), wh=r["wh"], max_vis=max_vis,
                         upscale=upscale, plen=(w // 32) * (h // 32) + 16, questions=[dict(q, type="box") for q in r["questions"]]))
        if limit and len(recs) >= limit:
            break
    return recs


def crop_window(W, H, cx, cy, frac, min_px=0):
    cw, ch = max(W * frac, min(W, min_px)), max(H * frac, min(H, min_px))
    x1 = min(max(0, cx - cw / 2), W - cw); y1 = min(max(0, cy - ch / 2), H - ch)
    return [x1, y1, x1 + cw, y1 + ch]


@torch.no_grad()
def run(net, heads, recs, cfg, topk, prefix_budget):
    ds = MixedDataset(None, "", net.cfg_proc, cfg, train=False, recs=recs)
    sampler = TokenBudgetSampler(ds, prefix_budget, 64, 16, shuffle=False, pool=len(ds) + 1)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=6,
                                     collate_fn=lambda it: collate_mixed(it, ds.tok))
    rows, maps = [], []
    t0 = time.time()
    for bn, batch in enumerate(dl):
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        feats = net.image_features(b["pixel_values"], b["grid_thw"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hp, hq = net(b, feats)
        M = hq.shape[0]
        hq_last = hq[torch.arange(M, device=hq.device), b["q_last"]]
        B, N = b["vis_valid"].shape
        vis = hp.new_zeros(B, N, hp.shape[-1])
        vis[b["vis_valid"]] = hp[b["p_img_mask"]]
        bh = heads.box
        out = bh(hq_last, vis, b["vis_valid"], b["centers"], b["q_img"])
        cl = out["center_logits"].float()
        lp = cl.log_softmax(-1)
        k = min(topk, N)
        tv, ti = lp.topk(k, -1)                                           # [M,k]
        # decode edges at each of the top-k patches
        hqn = bh.q_norm(hq_last.float()); visn = bh.v_norm(vis.float())
        qi = b["q_img"]
        v_sel = visn[qi[:, None].expand(-1, k), ti]                        # [M,k,D]
        pc = b["centers"][qi[:, None].expand(-1, k), ti]                   # [M,k,2]
        el = bh.edges_logits(hqn[:, None].expand(-1, k, -1).reshape(M * k, -1), v_sel.reshape(M * k, -1), pc.reshape(M * k, 2))
        bx = bh.expected_box(el, pc.reshape(M * k, 2)).view(M, k, 4).clamp(0, 1)
        piou = bh.iou(bh._edge_feats(hqn[:, None].expand(-1, k, -1).reshape(M * k, -1), v_sel.reshape(M * k, -1),
                                     pc.reshape(M * k, 2))).view(M, k).sigmoid()
        ent = -(lp.exp() * lp.masked_fill(lp < -1e3, 0)).sum(-1)
        pl = out["presence_logit"].float().cpu(); ghw = b["grid_hw"].cpu()
        tv, ti, pc, bx, piou, ent, lp = tv.cpu(), ti.cpu(), pc.cpu(), bx.cpu(), piou.cpu(), ent.cpu(), lp.cpu()
        m = 0
        for it in meta:
            W, H = it["orig_wh"]
            rec = it["rec"]
            ox, oy = (rec["crop"][0], rec["crop"][1]) if rec.get("crop") else (0.0, 0.0)
            for q in it["qs"]:
                gh, gw = int(ghw[m, 0]), int(ghw[m, 1])
                S = torch.tensor([W, H, W, H]); O = torch.tensor([ox, oy, ox, oy])
                cand = []
                for j in range(k):
                    bj = (bx[m, j] * S + O).tolist()
                    cand.append(dict(p=round(float(tv[m, j].exp()), 5), i=int(ti[m, j]),
                                     c=[round(float(pc[m, j, 0] * W + ox), 1), round(float(pc[m, j, 1] * H + oy), 1)],
                                     b=[round(v, 1) for v in bj], iou=round(float(piou[m, j]), 4)))
                r = dict(id=q["meta"]["id"], meta=q["meta"], text=q["text"], present=bool(q["present"]),
                         gt=q.get("gt_full", q.get("box")), wh=rec.get("full_wh", [W, H]), crop=rec.get("crop"),
                         grid=[gh, gw], p_logit=float(pl[m]), ent=round(float(ent[m]), 4), top=cand,
                         box=cand[0]["b"], p_iou=cand[0]["iou"])
                rows.append(r)
                maps.append(lp[m, :gh * gw].numpy().astype(np.float16))
                m += 1
        if bn % 20 == 0:
            print(f"[run] batch {bn}/{len(sampler)} rows {len(rows)} {time.time()-t0:.0f}s", flush=True)
    return rows, maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--max_vis", type=int, default=2048)
    ap.add_argument("--upscale", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefix_budget", type=int, default=20000)
    ap.add_argument("--zoom_from", default="", help="first-pass prefix (.rows.jsonl.gz): crop around its point")
    ap.add_argument("--zoom_frac", type=float, default=0.4)
    ap.add_argument("--zoom_min_px", type=float, default=0)
    ap.add_argument("--zoom_point", default="box", help="box | patch : which first-pass point to center the crop on")
    ap.add_argument("--zoom_cands", type=int, default=1, help="zoom on up to K distinct first-pass candidates")
    ap.add_argument("--no_maps", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update(sd["cfg"]).update(dict(grad_ckpt=False, max_q_tokens=96))
    net, heads, proc = build(cfg, "cuda")
    load_ckpt(a.ckpt, net, heads)
    net.cfg_proc = proc
    net.eval(); heads.eval()
    recs = load_bench(a.bench, a.max_vis, a.upscale, a.limit)
    if a.zoom_from:
        first = {}
        with gzip.open(a.zoom_from + ".rows.jsonl.gz", "rt") as f:
            for l in f:
                r = json.loads(l); first[r["id"]] = r
        zr = []
        for r in recs:
            W, H = r["wh"]
            for q in r["questions"]:
                fr = first[q["meta"]["id"]]
                if a.zoom_point == "patch":
                    pts = [(fr["top"][0]["c"], 1.0)]
                else:
                    bx = fr["box"]; pts = [([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2], fr["top"][0]["p"])]
                # --zoom_cands K: also zoom on the next-best first-pass candidates that fall outside the crops so far
                for c in fr["top"][1:]:
                    if len(pts) >= a.zoom_cands:
                        break
                    cx, cy = (c["b"][0] + c["b"][2]) / 2, (c["b"][1] + c["b"][3]) / 2
                    if all(abs(cx - p[0][0]) > W * a.zoom_frac / 2 or abs(cy - p[0][1]) > H * a.zoom_frac / 2 for p in pts):
                        pts.append(([cx, cy], c["p"]))
                for j, ((cx, cy), cp) in enumerate(pts):
                    crop = crop_window(W, H, cx, cy, a.zoom_frac, a.zoom_min_px)
                    cw_, ch_ = resize_dims(crop[2] - crop[0], crop[3] - crop[1], a.max_vis, a.upscale)
                    meta = dict(q["meta"], id=q["meta"]["id"] if j == 0 else f'{q["meta"]["id"]}#{j}', cand=j, cand_p=cp)
                    zr.append(dict(r, crop=crop, full_wh=[W, H], plen=(cw_ // 32) * (ch_ // 32) + 16,
                                   questions=[dict(q, box=None, gt_full=q.get("box"), meta=meta)]))
        recs = zr
    t0 = time.time()
    rows, maps = run(net, heads, recs, cfg, a.topk, a.prefix_budget)
    dt = time.time() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with gzip.open(a.out + ".rows.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not a.no_maps:
        np.savez_compressed(a.out + ".maps.npz", **{f"m{i}": x for i, x in enumerate(maps)})
    hit = [r["gt"][0] <= (r["box"][0] + r["box"][2]) / 2 <= r["gt"][2] and r["gt"][1] <= (r["box"][1] + r["box"][3]) / 2 <= r["gt"][3]
           for r in rows if r["present"] and r["gt"]]
    info = dict(out=a.out, n=len(rows), click_acc_box=float(np.mean(hit)) if hit else None, wall_s=dt,
                ckpt=a.ckpt, max_vis=a.max_vis, upscale=a.upscale, zoom_from=a.zoom_from, zoom_frac=a.zoom_frac,
                vis_tokens_median=float(np.median([r["grid"][0] * r["grid"][1] for r in rows])))
    json.dump(info, open(a.out + ".info.json", "w"), indent=1)
    print("[done]", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
