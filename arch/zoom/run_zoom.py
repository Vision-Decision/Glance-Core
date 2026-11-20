"""Native in-block zoom (zero training): coarse shared image prefix, then per question a crop of the full-resolution image
around the first-pass box is encoded by the vision tower and fed INSIDE that question's block; the box head re-predicts
over the crop's tokens. The shared prefix is never recomputed or modified, so any number of questions still share it.

  python -m arch.zoom.run_zoom --ckpt CKPT --bench /content/gui/sspro.jsonl --prefix_vis 16384 --crop_frac 0.25 \
      --crop_vis 4096 --crop_up 2 --layout pre --out DIR/name
Layouts of the zoom block (prefix = <im_start>user\\n<vs><img>*n<ve>):
  pre   <vs><crop>*m<ve> \\n<ref>{q}</ref><im_end>\\n<im_start>assistant\\n<box_start>   crop tokens causal (never see q);
        question tokens see the crop (and each other, as in training)
  post  \\n<ref>{q}</ref> <vs><crop>*m<ve> <im_end>\\n<im_start>assistant\\n<box_start>  whole block bidirectional (as trained):
        the crop tokens can attend to the question (question-conditioned crop encoding)
Also (--reprefill) the earlier two-pass baseline: a separate prefill of the crop alone + plain block (no full-image context).
Rows: DIR/name.rows.jsonl.gz; timings per image; summary DIR/name.json.
"""
import argparse, gzip, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from PIL import Image

from dv.config import Config
from dv.model.build import build, load_ckpt
from dv.data.mixed import ChatTok
from dv.data.dataset import resize_dims
from arch.zoom.lite import Lite, image_positions, grid_centers

Image.MAX_IMAGE_PIXELS = None


def sync():
    torch.cuda.synchronize()


class Zoom:
    def __init__(self, net, heads, proc, cfg):
        self.net, self.bh, self.proc, self.cfg = net, heads.box, proc, cfg
        self.tok = ChatTok(proc, cfg)
        self.E = Lite(net)

    def vit(self, imgs):
        pv = self.proc.image_processor(images=imgs, do_resize=False, return_tensors="pt")
        g = pv["image_grid_thw"]
        feats = self.net.image_features(pv["pixel_values"].cuda(), g.cuda())
        return feats, [(int(x[1]) // 2, int(x[2]) // 2) for x in g]

    def prefix(self, feat, gh, gw, keep=None):
        """keep: optional LongTensor of kept visual-token indices (question-independent reduction; original 2D positions kept)."""
        t = self.tok
        ids = list(t.user_open) + [t.vs]
        pos = [torch.arange(len(ids)).repeat(3, 1)]
        cur = len(ids)
        ip = image_positions(cur, gh, gw)
        n = gh * gw
        if keep is not None:
            ip = ip[:, keep]; feat = feat[keep.to(feat.device)]; n = len(keep)
        ids += [t.pad_img] * n; pos.append(ip)
        cur += max(gh, gw)
        ids.append(t.ve); pos.append(torch.tensor([cur]).repeat(3, 1)); cur += 1
        ids = torch.tensor([ids]).cuda(); pos = torch.cat(pos, 1)[:, None].cuda()
        img_mask = ids == t.pad_img
        cache, hp = self.E.prefill(ids, pos, feat, img_mask)
        vis = hp[0][img_mask[0]]
        return cache, vis, cur

    def plain_blocks(self, cache, q_start, texts):
        bl = [self.tok.box_block(x) for x in texts]
        T = max(map(len, bl)); M = len(bl)
        ids = torch.full((M, T), self.tok.pad, dtype=torch.long)
        valid = torch.zeros(M, T, dtype=torch.bool)
        for m, b in enumerate(bl):
            ids[m, :len(b)] = torch.tensor(b); valid[m, :len(b)] = True
        pos = (q_start + torch.arange(T)).repeat(3, M, 1)
        within = valid[:, None, :].expand(M, T, T) | torch.eye(T, dtype=torch.bool)[None]
        h = self.E.blocks(cache, ids.cuda(), pos.cuda(), within.cuda())
        last = torch.tensor([len(b) - 1 for b in bl]).cuda()
        return h[torch.arange(M).cuda(), last]

    def head(self, hq, vis_list, cen_list):
        """hq [M,D]; vis_list: one [n_i,D] per 'image'; returns head outputs with q_img = arange if len(vis_list)==M else 0."""
        B = len(vis_list); N = max(v.shape[0] for v in vis_list); D = hq.shape[-1]
        vis = hq.new_zeros(B, N, D); valid = torch.zeros(B, N, dtype=torch.bool, device=hq.device)
        cen = torch.zeros(B, N, 2, device=hq.device)
        for b, (v, c) in enumerate(zip(vis_list, cen_list)):
            vis[b, :len(v)] = v; valid[b, :len(v)] = True; cen[b, :len(v)] = c.to(hq.device)
        q_img = torch.arange(hq.shape[0], device=hq.device) if B == hq.shape[0] and B > 1 else torch.zeros(hq.shape[0], dtype=torch.long, device=hq.device)
        return self.bh(hq, vis, valid, cen, q_img)

    def zoom_blocks(self, cache, q_start, texts, crop_feats, crop_grids, layout, iso_img=None):
        """iso_img: (start, end) prefix index range of the full-image tokens; if given, the crop tokens may not attend to
        them in the full-attention layers (crop encoded as if alone; the question still sees both images)."""
        t = self.tok
        seqs, poss, withins, imgm = [], [], [], []
        for x, (gh, gw) in zip(texts, crop_grids):
            n = gh * gw
            q_ids = t.enc(x)[: self.cfg.max_q_tokens]
            qtxt = t.enc("\n") + [t.ors] + q_ids + [t.ore]
            if layout in ("pre", "pre_iso", "pre_alliso"):
                segs = [("c", [t.vs]), ("img", n), ("c", [t.ve]), ("q", qtxt + t.asst_box)]
            else:
                segs = [("q", qtxt), ("c", [t.vs]), ("img", n), ("c", [t.ve]), ("q", t.asst_box)]
            ids, pos, kind = [], [], []
            cur = q_start
            for k, s in segs:
                if k == "img":
                    ids += [t.pad_img] * s; pos.append(image_positions(cur, gh, gw)); kind += ["c"] * s; cur += max(gh, gw)
                else:
                    ids += s; pos.append((cur + torch.arange(len(s))).repeat(3, 1)); kind += [k] * len(s); cur += len(s)
            seqs.append(ids); poss.append(torch.cat(pos, 1)); withins.append(kind)
        M = len(seqs); T = max(map(len, seqs))
        ids = torch.full((M, T), t.pad, dtype=torch.long); pos = torch.zeros(3, M, T, dtype=torch.long)
        within = torch.zeros(M, T, T, dtype=torch.bool)
        for m, (s, p, kind) in enumerate(zip(seqs, poss, withins)):
            L = len(s)
            ids[m, :L] = torch.tensor(s); pos[:, m, :L] = p; pos[:, m, L:] = p[:, -1:] + 1 + torch.arange(T - L)
            isc = torch.tensor([k == "c" for k in kind])
            if layout in ("pre", "pre_iso", "pre_alliso"):
                causal = torch.ones(L, L, dtype=torch.bool).tril()
                w = (isc[:, None] & isc[None, :] & causal) | (~isc)[:, None]
            else:
                w = torch.ones(L, L, dtype=torch.bool)
            within[m, :L, :L] = w
        within |= torch.eye(T, dtype=torch.bool)[None]
        img_mask = ids == t.pad_img
        pmt = None
        if iso_img is not None:
            Lp = next(c[1].shape[2] for c in cache if c[0] == "attn")
            pmt = torch.ones(M, T, Lp, dtype=torch.bool)
            for m, kind in enumerate(withins):
                isc = torch.tensor([k == "c" for k in kind] + [False] * (T - len(kind)))
                if layout == "pre_alliso":
                    isc[:] = True
                pmt[m, isc, iso_img[0]:iso_img[1]] = False
        h = self.E.blocks(cache, ids.cuda(), pos.cuda(), within.cuda(), torch.cat(crop_feats, 0), img_mask.cuda(), prefix_mask_tok=pmt)
        last = torch.tensor([len(s) - 1 for s in seqs]).cuda()
        hq = h[torch.arange(M).cuda(), last]
        vis = [h[m][img_mask[m].cuda()] for m in range(M)]
        return hq, vis


def crop_window(W, H, cx, cy, frac):
    cw, ch = W * frac, H * frac
    x1 = min(max(0, cx - cw / 2), W - cw); y1 = min(max(0, cy - ch / 2), H - ch)
    return [x1, y1, x1 + cw, y1 + ch]


def inside(b, g):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return bool(g[0] <= cx <= g[2] and g[1] <= cy <= g[3])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", default="/content/gui/sspro.jsonl")
    ap.add_argument("--prefix_vis", type=int, default=16384)
    ap.add_argument("--crop_frac", type=float, default=0.25)
    ap.add_argument("--crop_vis", type=int, default=4096)
    ap.add_argument("--crop_up", type=float, default=2.0)
    ap.add_argument("--layout", default="pre", help="pre | post | none (first pass only)")
    ap.add_argument("--reprefill", action="store_true", help="also run the separate-prefill crop baseline")
    ap.add_argument("--limit", type=int, default=0, help="images")
    ap.add_argument("--stride", type=int, default=1, help="take every k-th image (subsets)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--cands", type=int, default=1, help="crops per question: first-pass box centre + next distinct top "
                    "centre patches; the crop whose in-block presence logit is highest answers")
    ap.add_argument("--alpha", type=float, default=1.0, help="LoRA-delta scale (exp 1) for the whole run")
    ap.add_argument("--keep_ratio", type=float, default=1.0, help="exp 8: keep this fraction of the prefix visual tokens "
                    "(question-independent: highest pixel std of the token's 32x32 patch), original M-RoPE positions kept")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__}).update(dict(grad_ckpt=False, max_q_tokens=96))
    net, heads, proc = build(cfg, "cuda"); load_ckpt(a.ckpt, net, heads); net.eval(); heads.eval()
    if a.alpha != 1.0:
        from arch.common import set_lora_scale
        set_lora_scale(net, a.alpha)
    Z = Zoom(net, heads, proc, cfg)
    recs = [json.loads(l) for l in open(a.bench)][:: a.stride]
    if a.limit:
        recs = recs[: a.limit]
    rows, T = [], dict(vit=0.0, prefill=0.0, stage1=0.0, vit_crop=0.0, stage2=0.0, reprefill=0.0)
    nq_timed = 0
    t_all = time.time()
    for ri, r in enumerate(recs):
        img = Image.open(os.path.join(r["image_root"], r["file"])).convert("RGB")
        W, H = img.size
        qs = r["questions"]; texts = [q["text"] for q in qs]
        timed = ri >= a.warmup
        sync(); t0 = time.perf_counter()
        w, h = resize_dims(W, H, a.prefix_vis, 1.0)
        rimg = img.resize((w, h), Image.BICUBIC)
        feats, grids = Z.vit([rimg])
        sync(); t1 = time.perf_counter()
        gh, gw = grids[0]
        keep, cen = None, grid_centers(gh, gw)
        if a.keep_ratio < 1.0:
            arr = np.asarray(rimg, dtype=np.float32).reshape(gh, 32, gw, 32, 3)
            sd_ = arr.std(axis=(1, 3, 4)).reshape(-1)
            keep = torch.from_numpy(np.sort(np.argsort(-sd_, kind="stable")[: max(1, int(round(a.keep_ratio * gh * gw)))]))
            cen = cen[keep]
        cache, vis, q_start = Z.prefix(feats[0], gh, gw, keep=keep)
        sync(); t2 = time.perf_counter()
        hq = Z.plain_blocks(cache, q_start, texts)
        o1 = Z.head(hq, [vis], [cen])
        b1 = (o1["box_pred"].clamp(0, 1).float().cpu() * torch.tensor([W, H, W, H])).tolist()
        sync(); t3 = time.perf_counter()
        out = [dict(id=q["meta"]["id"], meta=q["meta"], gt=q["box"], wh=[W, H], n_prefix=len(vis),
                    box1=[round(v, 1) for v in b1[m]], p1=float(o1["presence_logit"][m]), k1=float(o1["iou_logit"][m].sigmoid()))
               for m, q in enumerate(qs)]
        if a.layout != "none":
            crops, cimgs, owner = [], [], []
            cl = o1["center_logits"].float().softmax(-1).cpu()
            cen_px = grid_centers(gh, gw) if keep is None else grid_centers(gh, gw)[keep]
            for m in range(len(qs)):
                pts = [((b1[m][0] + b1[m][2]) / 2, (b1[m][1] + b1[m][3]) / 2)]
                for i in cl[m].argsort(descending=True)[:64].tolist():
                    if len(pts) >= a.cands:
                        break
                    x, y = float(cen_px[i, 0]) * W, float(cen_px[i, 1]) * H
                    if all(abs(x - px) > W * a.crop_frac / 2 or abs(y - py) > H * a.crop_frac / 2 for px, py in pts):
                        pts.append((x, y))
                for cx, cy in pts:
                    c = crop_window(W, H, cx, cy, a.crop_frac); crops.append(c); owner.append(m)
                    ci = img.crop(tuple(int(round(v)) for v in c))
                    cw_, ch_ = resize_dims(ci.size[0], ci.size[1], a.crop_vis, a.crop_up)
                    cimgs.append(ci.resize((cw_, ch_), Image.BICUBIC))
            sync(); t4 = time.perf_counter()
            cf, cg = Z.vit(cimgs)
            sync(); t5 = time.perf_counter()
            n0 = len(Z.tok.user_open) + 1
            hq2, vis2 = Z.zoom_blocks(cache, q_start, [texts[m] for m in owner], cf, cg, a.layout,
                                      iso_img=(n0, n0 + len(vis)) if a.layout in ("pre_iso", "pre_alliso") else None)
            o2 = Z.head(hq2, vis2, [grid_centers(*g) for g in cg])
            bx = o2["box_pred"].clamp(0, 1).float().cpu()
            sync(); t6 = time.perf_counter()
            cand = [[] for _ in qs]
            for j, (m, c) in enumerate(zip(owner, crops)):
                cw, ch = c[2] - c[0], c[3] - c[1]
                b2 = [c[0] + float(bx[j, 0]) * cw, c[1] + float(bx[j, 1]) * ch, c[0] + float(bx[j, 2]) * cw, c[1] + float(bx[j, 3]) * ch]
                cand[m].append(dict(crop=[round(v, 1) for v in c], n_crop=cg[j][0] * cg[j][1], box2=[round(v, 1) for v in b2],
                                    p2=float(o2["presence_logit"][j]), k2=float(o2["iou_logit"][j].sigmoid())))
            for m in range(len(qs)):
                best = max(cand[m], key=lambda d: d["p2"])
                out[m].update(**best)
                if a.cands > 1:
                    out[m]["cands"] = cand[m]
                    out[m]["hit2_first"] = inside(cand[m][0]["box2"], out[m]["gt"])
                    out[m]["hit2_any"] = any(inside(d["box2"], out[m]["gt"]) for d in cand[m])
            if timed:
                T["vit_crop"] += t5 - t4; T["stage2"] += t6 - t5
            if a.reprefill:
                sync(); t7 = time.perf_counter()
                for m, c in enumerate(crops[: len(qs)] if a.cands == 1 else []):
                    g = cg[m]
                    cache_c, vis_c, qs_c = Z.prefix(cf[m], *g)
                    hq_c = Z.plain_blocks(cache_c, qs_c, [texts[m]])
                    oc = Z.head(hq_c, [vis_c], [grid_centers(*g)])
                    bc = oc["box_pred"].clamp(0, 1).float().cpu()[0]
                    cw, ch = c[2] - c[0], c[3] - c[1]
                    out[m].update(box_rp=[round(c[0] + float(bc[0]) * cw, 1), round(c[1] + float(bc[1]) * ch, 1),
                                          round(c[0] + float(bc[2]) * cw, 1), round(c[1] + float(bc[3]) * ch, 1)],
                                  p_rp=float(oc["presence_logit"][0]))
                sync()
                if timed:
                    T["reprefill"] += time.perf_counter() - t7
        if timed:
            T["vit"] += t1 - t0; T["prefill"] += t2 - t1; T["stage1"] += t3 - t2; nq_timed += len(qs)
        for o in out:
            o["hit1"] = inside(o["box1"], o["gt"])
            if "box2" in o:
                o["hit2"] = inside(o["box2"], o["gt"])
            if "box_rp" in o:
                o["hit_rp"] = inside(o["box_rp"], o["gt"])
        rows += out
        if ri % 50 == 0:
            print(f"[zoom] {ri}/{len(recs)} rows {len(rows)} hit1 {np.mean([x['hit1'] for x in rows]):.3f}"
                  + (f" hit2 {np.mean([x['hit2'] for x in rows]):.3f}" if "hit2" in rows[0] else "") + f" {time.time()-t_all:.0f}s", flush=True)
        del cache
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with gzip.open(a.out + ".rows.jsonl.gz", "wt") as f:
        for o in rows:
            f.write(json.dumps(o) + "\n")
    S = dict(vars(a), n=len(rows), n_images=len(recs), acc1=float(np.mean([x["hit1"] for x in rows])),
             ms_per_q={k: 1000 * v / max(1, nq_timed) for k, v in T.items()}, nq_timed=nq_timed,
             prefix_tok_median=float(np.median([x["n_prefix"] for x in rows])), wall_s=time.time() - t_all)
    if "hit2" in rows[0]:
        S["acc2"] = float(np.mean([x["hit2"] for x in rows]))
        S["crop_tok_median"] = float(np.median([x["n_crop"] for x in rows]))
    if "hit2_any" in rows[0]:
        S["acc2_first_crop"] = float(np.mean([x["hit2_first"] for x in rows])); S["acc2_any_crop"] = float(np.mean([x["hit2_any"] for x in rows]))
    if "hit_rp" in rows[0]:
        S["acc_rp"] = float(np.mean([x["hit_rp"] for x in rows]))
    json.dump(S, open(a.out + ".json", "w"), indent=1)
    print("[zoom] DONE", json.dumps(S), flush=True)


if __name__ == "__main__":
    main()
