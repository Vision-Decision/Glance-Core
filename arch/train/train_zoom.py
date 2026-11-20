"""Phase 2: train the in-block zoom step (from v1.1 LoRA + heads). Each optimizer step = 1 zoom micro-batch
(arch/train/zoom_data.py; box loss over the crop's tokens) + 1 replay micro-batch of the v1.1 mixture (standard loss),
so the first pass and the typed heads are kept.

  python -m arch.train.train_zoom --init /content/ckpt/.../v1.1/ckpt_final.pt --train /content/data/prep_arch/train_base_avail.jsonl \
      --steps 800 --out /content/out/arch/zoom_t1
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from dv.config import Config
from dv.model.build import build, load_ckpt, save_ckpt
from dv.model.heads import box_loss_v1, center_target_box
from train import run_model_v1, loss_v1, make_loader_v1, to_dev
from arch.train.zoom_data import ZoomDataset, collate_zoom


def zoom_loss(net, heads, b, cfg):
    feats = net.image_features(b["pixel_values"], b["grid_thw"])
    qf = net.image_features(b["crop_pixel_values"], b["crop_grid_thw"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hp, hq = net(b, feats, q_feats=qf)
    M, T, D = hq.shape
    hq_last = hq[torch.arange(M, device=hq.device), b["q_last"]]
    N = b["vis_valid"].shape[1]
    vis = hq.new_zeros(M, N, D)
    vis[b["vis_valid"]] = hq[b["q_img_mask"]]
    q_img = torch.arange(M, device=hq.device)
    ct = center_target_box(b["box"], b["centers"], b["vis_valid"], b["grid_hw"]) * b["has_box"][:, None]
    noct = ct.sum(-1) <= 0
    sel = torch.multinomial(ct + noct[:, None] * b["vis_valid"].float() + 1e-12, 1)[:, 0]
    out = heads.box(hq_last, vis, b["vis_valid"], b["centers"], q_img, sel_idx=sel)
    loss, parts = box_loss_v1(out, dict(present=b["present"], has_box=b["has_box"], box=b["box"], center_t=ct), cfg)
    with torch.no_grad():
        bp, g = out["box_pred"].float(), b["box"]
        cx, cy = (bp[:, 0] + bp[:, 2]) / 2, (bp[:, 1] + bp[:, 3]) / 2
        hit = (cx >= g[:, 0]) & (cx <= g[:, 2]) & (cy >= g[:, 1]) & (cy <= g[:, 3])
        parts["_click"] = hit[b["has_box"]].float().mean() if b["has_box"].any() else hit.float().sum() * 0
    return loss, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--data_root", default="/content/data")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_heads", type=float, default=2e-4)
    ap.add_argument("--zoom_bs", type=int, default=6, help="records per zoom micro-batch (q_per_rec blocks each)")
    ap.add_argument("--prefix_vis", type=int, default=2048)
    ap.add_argument("--crop_vis", type=int, default=1024)
    ap.add_argument("--w_zoom", type=float, default=1.0)
    ap.add_argument("--no_replay", action="store_true")
    ap.add_argument("--no_iso", action="store_true", help="train the plain 'pre' layout (crop sees the full image)")
    ap.add_argument("--freeze_lora", action="store_true", help="heads only (no backbone change: no extra forgetting)")
    ap.add_argument("--alpha", type=float, default=1.0, help="LoRA-delta scale during training (and to use at test time)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sd = torch.load(a.init, map_location="cpu", weights_only=False)
    cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__}).update(dict(train_file=a.train, data_root=a.data_root, grad_ckpt=True, out_dir=a.out,
                                                 max_q_tokens=96))
    torch.manual_seed(0)
    net, heads, proc = build(cfg, "cuda"); load_ckpt(a.init, net, heads); net.train(); heads.train()
    zds = ZoomDataset(a.train, a.data_root, proc, cfg, prefix_vis=a.prefix_vis, crop_vis=a.crop_vis, iso=not a.no_iso)
    print(f"[zoom] {len(zds)} GUI records", flush=True)
    zdl = torch.utils.data.DataLoader(zds, batch_size=a.zoom_bs, shuffle=True, num_workers=6, drop_last=True,
                                      collate_fn=lambda it: collate_zoom(it, zds.tok), persistent_workers=True)
    rdl = None if a.no_replay else make_loader_v1(cfg, proc)[2]
    if a.alpha != 1.0:
        from arch.common import set_lora_scale
        set_lora_scale(net, a.alpha)
    if a.freeze_lora:
        for p in net.hf.parameters():
            p.requires_grad_(False)
    lora_p = [p for p in net.hf.parameters() if p.requires_grad]
    groups = ([dict(params=lora_p, lr=a.lr)] if lora_p else []) + [dict(params=list(heads.parameters()), lr=a.lr_heads)]
    opt = torch.optim.AdamW(groups,
                            betas=(0.9, 0.95), fused=True)
    base = [g["lr"] for g in opt.param_groups]
    zit, rit = iter(zdl), (iter(rdl) if rdl else None)
    log = open(f"{a.out}/train_log.jsonl", "a"); t0 = time.time()
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    for step in range(a.steps):
        f = min(1.0, (step + 1) / 50) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * f
        try:
            zb = next(zit)
        except StopIteration:
            zit = iter(zdl); zb = next(zit)
        zl, zp = zoom_loss(net, heads, to_dev(zb, "cuda"), cfg)
        (a.w_zoom * zl).backward()
        rec = dict(step=step + 1, zoom=float(zl), zoom_acc50=float(zp.get("_acc50", 0)), zoom_pres=float(zp["presence"]), zoom_click=float(zp["_click"]))
        if rit is not None:
            try:
                rb = next(rit)
            except StopIteration:
                rit = iter(rdl); rb = next(rit)
            res = run_model_v1(net, heads, to_dev(rb, "cuda"), train=True)
            rl, _ = loss_v1(res, to_dev(rb, "cuda"), cfg)
            if torch.is_tensor(rl):
                rl.backward(); rec["replay"] = float(rl)
        gn = torch.nn.utils.clip_grad_norm_(lora_p + list(heads.parameters()), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 100 == 0:
            save_ckpt(f"{a.out}/ckpt_last.pt", net, heads, cfg, dict(step=step + 1))
        if step % 10 == 0:
            rec.update(gnorm=float(gn), min=(time.time() - t0) / 60, mem=torch.cuda.max_memory_allocated() / 2**30)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    save_ckpt(f"{a.out}/ckpt_final.pt", net, heads, cfg, dict(step=a.steps, zoom_args=vars(a)))
    print("[zoom] DONE", flush=True)


if __name__ == "__main__":
    main()
