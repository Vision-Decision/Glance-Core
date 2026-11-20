"""Absence/confidence sprint: dump extended per-question rows for a v1-style checkpoint (VM, GPU).

python scripts/sprint_abs/dump.py --ckpt /content/ckpt/v1/ckpt_final.pt --out /content/sabs/v1 \
    --sets calib_natural,grefcalib,guisyn,refcoco_val,...,grec_val,grec_testA,grec_testB,ss2,venus

Per box question (gzip JSONL, one file per set) on top of eval.py's fields:
  p_logit (presence logit, raw), iou_logit (kappa logit, raw), center_top1 / center_ent (center distribution),
  edge_sd (std of the 4 edge distributions, normalized units), n_vis (visual tokens),
  cand: top-K center patches -> [p_center, mass_in_box, iou_logit, x1, y1, x2, y2 (px)] with the box decoded at that patch
        and mass_in_box = center probability mass of all patches whose centers fall inside that box
        (used offline for multi-target GREC decoding; K=16).
Nothing is fitted here; all calibration / decision rules are fitted offline on the calib_* / grefcalib / guisyn-fit rows.
"""
import argparse, gzip, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dv.config import Config  # noqa: E402
from dv.model.build import build, load_ckpt  # noqa: E402
from dv.model.heads import box_iou_giou  # noqa: E402
from train import to_dev  # noqa: E402
from eval import v0_eval_to_v1, gui_to_v1, merged_grefcoco  # noqa: E402


def grec_split(prep_dir, sp):
    """gRefCOCO <sp> as released (testA/testB contain no-, single- and multi-target expressions)."""
    out = []
    for l in open(f"{prep_dir}/eval_grefcoco_{sp}.jsonl"):
        r = json.loads(l)
        out.append(dict(src="eval", img=f"coco/train2014/{r['file']}", wh=[r["W"], r["H"]], max_vis=640, plen=0,
                        questions=[dict(q, type="box", n_targets=q.get("n_targets", 1) if q["present"] else 0,
                                        boxes=q.get("boxes") or ([q["box"]] if q.get("box") else [])) for q in r["questions"]]))
    return out


def load_jsonl(fn):
    return [json.loads(l) for l in open(fn)]


@torch.no_grad()
def predict_ext(net, heads, recs, data_root, cfg, K=16, prefix_budget=16384, q_budget=256):
    from dv.data.mixed import MixedDataset, collate_mixed, TokenBudgetSampler
    ds = MixedDataset(None, data_root, net.cfg_proc, cfg, train=False, recs=recs)
    tok = ds.tok
    sampler = TokenBudgetSampler(ds, prefix_budget, q_budget, 64, shuffle=False, pool=len(ds) + 1)
    # large GUI prefixes: few workers, or prefetched pixel tensors exhaust /dev/shm (40 GB) and the loader hangs
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=8 if prefix_budget > 2048 else 2,
                                     collate_fn=lambda it: collate_mixed(it, tok))
    H = heads.box
    rows = []
    for batch in dl:
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        feats = net.image_features(b["pixel_values"], b["grid_thw"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hp, hq = net(b, feats)
        M = hq.shape[0]
        hq_last = hq[torch.arange(M, device=hq.device), b["q_last"]]
        is_box = b["qtype"] == 0
        bi = is_box.nonzero()[:, 0]
        Bn, N = b["vis_valid"].shape
        vis = hp.new_zeros(Bn, N, hp.shape[-1])
        vis[b["vis_valid"]] = hp[b["p_img_mask"]]
        q_img = b["q_img"][bi]
        out = H(hq_last[bi], vis, b["vis_valid"], b["centers"], q_img)
        m = len(bi)
        cprob = out["center_logits"].float().softmax(-1)
        ent = -(cprob * (cprob + 1e-12).log()).sum(-1)
        Kk = min(K, N)
        topv, topi = cprob.topk(Kk, -1)
        hqn = H.q_norm(hq_last[bi].float()); visn = H.v_norm(vis.float())
        D = hqn.shape[-1]
        cen = b["centers"][q_img].float()  # [m,N,2]
        hq_rep = hqn[:, None].expand(m, Kk, D).reshape(-1, D)
        v_sel = visn[q_img[:, None].expand(m, Kk).reshape(-1), topi.reshape(-1)]
        pc = cen.gather(1, topi[..., None].expand(m, Kk, 2)).reshape(-1, 2)
        el = H.edges_logits(hq_rep, v_sel, pc)
        cboxes = H.expected_box(el, pc).view(m, Kk, 4).clamp(0, 1)
        ciou = H.iou(H._edge_feats(hq_rep, v_sel, pc)).squeeze(-1).view(m, Kk)
        cx, cy = cen[:, None, :, 0], cen[:, None, :, 1]
        inside = (cx >= cboxes[..., 0:1]) & (cx <= cboxes[..., 2:3]) & (cy >= cboxes[..., 1:2]) & (cy <= cboxes[..., 3:4])
        mass = (cprob[:, None, :] * inside).sum(-1)
        # peaks: repeatedly take the most probable remaining patch, decode its box, suppress all patches inside it
        Pm = cprob.clone()
        ar = torch.arange(m, device=Pm.device)
        pk = []
        for _ in range(6):
            pv, pidx = Pm.max(-1)
            vs, pcs = visn[q_img, pidx], cen[ar, pidx]
            elp = H.edges_logits(hqn, vs, pcs)
            bxp = H.expected_box(elp, pcs).clamp(0, 1)
            ilp = H.iou(H._edge_feats(hqn, vs, pcs)).squeeze(-1)
            ins = (cen[..., 0] >= bxp[:, 0:1]) & (cen[..., 0] <= bxp[:, 2:3]) & (cen[..., 1] >= bxp[:, 1:2]) & (cen[..., 1] <= bxp[:, 3:4])
            pk.append(torch.cat([pv[:, None], (cprob * ins).sum(-1)[:, None], (Pm * ins).sum(-1)[:, None], ilp[:, None].float(), bxp], -1))
            Pm = Pm.masked_fill(ins, 0); Pm[ar, pidx] = 0
        peaks = torch.stack(pk, 1).cpu()  # [m,6,8]
        pe = out["edge_logits_pred"].float().softmax(-1)
        mu = (pe * H.bin_centers).sum(-1)
        sd = ((pe * (H.bin_centers - mu[..., None]) ** 2).sum(-1)).sqrt()
        nvis = b["vis_valid"][q_img].sum(-1)
        res = dict(pl=out["presence_logit"].float().cpu(), il=out["iou_logit"].float().cpu(),
                   box=out["box_pred"].clamp(0, 1).float().cpu(), top1=topv[:, 0].cpu(), ent=ent.cpu(),
                   tv=topv.cpu(), cb=cboxes.cpu(), ci=ciou.float().cpu(), cm=mass.cpu(), sd=sd.cpu(), nv=nvis.cpu())
        pos = {int(x): j for j, x in enumerate(bi.cpu().tolist())}
        k = 0
        for mt in meta:
            W, Hh = mt["orig_wh"]
            rec = mt["rec"]
            for q in mt["qs"]:
                if k in pos:
                    j = pos[k]
                    sc = torch.tensor([W, Hh, W, Hh])
                    bx = res["box"][j] * sc
                    r = dict(text=q["text"], present=bool(q["present"]), p_logit=float(res["pl"][j]),
                             iou_logit=float(res["il"][j]), box=[round(float(v), 1) for v in bx],
                             n_targets=q.get("n_targets", 1 if q["present"] else 0), wh=[W, Hh], neg=q.get("neg"),
                             qsrc=q.get("src"), meta=q.get("meta"), sub=rec.get("sub"), img=rec.get("img"),
                             center_top1=round(float(res["top1"][j]), 5), center_ent=round(float(res["ent"][j]), 4),
                             edge_sd=[round(float(v), 5) for v in res["sd"][j]], n_vis=int(res["nv"][j]),
                             gt_box=q.get("box"), gt_boxes=q.get("boxes"))
                    cand = []
                    for c in range(Kk):
                        cb = (res["cb"][j, c] * sc).tolist()
                        cand.append([round(float(res["tv"][j, c]), 5), round(float(res["cm"][j, c]), 5),
                                     round(float(res["ci"][j, c]), 4)] + [round(v, 1) for v in cb])
                    r["cand"] = cand
                    r["peaks"] = [[round(float(v), 5) for v in peaks[j, c, :4]] + [round(v, 1) for v in (peaks[j, c, 4:] * sc).tolist()]
                                  for c in range(peaks.shape[1])]
                    if q.get("box") is not None:
                        iou, giou = box_iou_giou(bx[None], torch.tensor(q["box"], dtype=torch.float)[None])
                        r["iou"] = float(iou[0]); r["giou"] = float(giou[0])
                        g = q["box"]; ccx, ccy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                        r["center_in_box"] = bool(g[0] <= ccx <= g[2] and g[1] <= ccy <= g[3])
                    rows.append(r)
                k += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--prep_dir", default="/content/dv/data/prep")
    ap.add_argument("--calib_dir", default="/content/sabs/data")
    ap.add_argument("--gui_dir", default="/content/gui")
    ap.add_argument("--gui_max_vis", type=int, default=2048)
    ap.add_argument("--gui_upscale", type=float, default=2.0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update(sd["cfg"]).update(dict(s.split("=", 1) for s in a.set), from_str=True)
    del sd
    net, heads, proc = build(cfg, "cuda")
    load_ckpt(a.ckpt, net, heads)
    net.cfg_proc = proc
    net.eval(); heads.eval()
    os.makedirs(a.out, exist_ok=True)
    root = cfg.data_root
    for name in a.sets.split(","):
        fn = f"{a.out}/{name}.jsonl.gz"
        if os.path.exists(fn):
            print("exists", fn); continue
        t0 = time.time()
        dr = root
        if name in ("calib_natural", "grefcalib"):
            recs = load_jsonl(f"{a.calib_dir}/{name}.jsonl")
        elif name == "guisyn":
            recs = load_jsonl(f"{a.calib_dir}/guisyn_calib.jsonl")
            for r in recs:
                r["max_vis"], r["upscale"] = a.gui_max_vis, a.gui_upscale
        elif name in ("ss2", "sspro", "venus"):
            recs = gui_to_v1(f"{a.gui_dir}/{name}.jsonl", a.gui_max_vis, a.gui_upscale); dr = ""
        elif name == "grec_val":
            recs = merged_grefcoco(a.prep_dir, "val")
        elif name.startswith("grec_"):
            recs = grec_split(a.prep_dir, name.split("_", 1)[1])
        else:
            recs = v0_eval_to_v1(f"{a.prep_dir}/eval_{name}.jsonl")
        rows = predict_ext(net, heads, recs, dr, cfg, prefix_budget=2048 if name in ("ss2", "sspro", "venus", "guisyn") else 16384)
        with gzip.open(fn + ".tmp", "wt") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        os.replace(fn + ".tmp", fn)
        print(f"[dump] {name} rows={len(rows)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
