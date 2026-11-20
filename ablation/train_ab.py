"""Train one ablation variant.  python -m ablation.train_ab --cfg ablation/configs/<variant>.json [--set k=v ...]

Same optimizer / schedule / LoRA / data order (dedicated loader generator, cfg.data_seed) / steps for every variant.
ours: frozen snapshot of the v0 head + losses (ablation/dvcore).  ar_*: token-mean CE over answer tokens.
"""
import json, math, os, random, time
import torch
import torch.nn.functional as F

from ablation.common import ABConfig, ARDataset, collate_ar, make_loader, build_ar, ar_state
from ablation.dvcore.dataset import RecordDataset, collate
from ablation.dvcore.heads import center_target, box_loss


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_ours(net, heads, batch, train=True):
    """Copy of train.run_model (v0)."""
    feats = net.image_features(batch["pixel_values"], batch["grid_thw"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hp, hq = net(batch, feats)
    M = hq.shape[0]
    hq_last = hq[torch.arange(M, device=hq.device), batch["q_last"]]
    B, N = batch["vis_valid"].shape
    vis = hp.new_zeros(B, N, hp.shape[-1])
    vis[batch["vis_valid"]] = hp[batch["p_img_mask"]]
    q_img = batch["q_img"]
    ct = sel = None
    if train:
        ct = center_target(batch["box"], batch["centers"][q_img], batch["vis_valid"][q_img], batch["grid_hw"], net.cfg)
        sel = torch.multinomial(ct + (~batch["present"])[:, None] * batch["vis_valid"][q_img].float() + 1e-12, 1)[:, 0]
    out = heads(hq_last, vis, batch["vis_valid"], batch["centers"], q_img, sel_idx=sel)
    return out, ct


def ar_loss(net, batch):
    feats = net.image_features(batch["pixel_values"], batch["grid_thw"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, hq = net(batch, feats)
        lab = batch["labels"]
        sel = lab != -100
        lg = net.logits(hq[sel])
    tgt = lab[sel]
    loss = F.cross_entropy(lg.float(), tgt)
    with torch.no_grad():
        # first-token stats (position = <box_start>)
        M = lab.shape[0]
        first = lab[torch.arange(M, device=lab.device), batch["q_last"]]
        is_first = torch.zeros_like(sel)
        is_first[torch.arange(M, device=lab.device), batch["q_last"]] = True
        fl = lg[is_first[sel]].float()
        pn = fl.softmax(-1)[:, net.fmt.none]
        pres = first != net.fmt.none
        parts = dict(tok_acc=(lg.argmax(-1) == tgt).float().mean(),
                     first_acc=(fl.argmax(-1) == first).float().mean(),
                     pnone_pos=pn[pres].mean() if pres.any() else pn.sum() * 0,
                     pnone_neg=pn[~pres].mean() if (~pres).any() else pn.sum() * 0,
                     ans_tokens=sel.sum().float())
    return loss, parts


def main():
    cfg, _ = ABConfig.parse()
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    open(f"{cfg.out_dir}/config.json", "w").write(cfg.to_json())
    dev = "cuda"
    t0 = time.time()
    if cfg.variant == "ours":
        from ablation.dvcore.build import build
        net, heads, proc = build(cfg, dev)
        heads.train()
        ds = RecordDataset(cfg.train_file, cfg.image_root, proc, cfg, limit=cfg.limit_records)
        tok = ds.tok
        cfn = lambda it: collate(it, tok)
        extra_params = list(heads.parameters())
    else:
        net, fmt, proc = build_ar(cfg, dev)
        heads = None
        ds = ARDataset(cfg.train_file, cfg.image_root, proc, cfg, limit=cfg.limit_records, fmt=fmt)
        tok = ds.tok
        cfn = lambda it: collate_ar(it, tok)
        extra_params = [net.new_emb] if net.new_emb is not None else []
    net.train()
    print(f"[load] {time.time()-t0:.1f}s variant={cfg.variant}", flush=True)
    dl = make_loader(ds, cfn, cfg)
    steps_per_epoch = math.ceil(len(dl) / cfg.grad_accum)
    total = cfg.max_steps or int(steps_per_epoch * cfg.epochs)
    lora_p = [p for n, p in net.hf.named_parameters() if p.requires_grad]
    print(f"[params] lora={sum(p.numel() for p in lora_p)/1e6:.1f}M extra={sum(p.numel() for p in extra_params)/1e6:.2f}M "
          f"records={len(ds)} steps={total}", flush=True)
    groups = [dict(params=lora_p, lr=cfg.lr)]
    if extra_params:
        groups.append(dict(params=extra_params, lr=cfg.lr_heads if cfg.variant == "ours" else cfg.lr_newtok))
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95), fused=True)
    base_lrs = [g["lr"] for g in opt.param_groups]
    all_p = lora_p + extra_params

    def lr_factor(s):
        if s < cfg.warmup_steps:
            return (s + 1) / cfg.warmup_steps
        p = (s - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    def save(path, step):
        if cfg.variant == "ours":
            from ablation.dvcore.build import save_ckpt
            save_ckpt(path, net, heads, cfg, dict(step=step))
        else:
            torch.save(dict(**ar_state(net), cfg=cfg.__dict__, extra=dict(step=step)), path)

    logf = open(f"{cfg.out_dir}/train_log.jsonl", "a")
    torch.cuda.reset_peak_memory_stats()
    step, micro, acc, tok_acc, t_step = 0, 0, {}, 0, time.time()
    t_train = time.time()
    done = False
    while not done:
        for batch in dl:
            batch.pop("meta", None)
            batch = to_dev(batch, dev)
            if cfg.variant == "ours":
                out, ct = run_ours(net, heads, batch, train=True)
                loss, parts = box_loss(out, dict(present=batch["present"], box=batch["box"], center_t=ct), cfg)
            else:
                loss, parts = ar_loss(net, batch)
            (loss / cfg.grad_accum).backward()
            micro += 1
            tok_acc += batch["n_tokens"]
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
            acc["loss"] = acc.get("loss", 0.0) + float(loss)
            acc["n_q"] = acc.get("n_q", 0) + len(batch["present"])
            if micro % cfg.grad_accum:
                continue
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_factor(step)
            gn = torch.nn.utils.clip_grad_norm_(all_p, cfg.clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                torch.cuda.synchronize()
                dt = time.time() - t_step
                n_micro = cfg.grad_accum * (cfg.log_every if step > 1 else 1)
                rec = {k: v / n_micro for k, v in acc.items() if k != "n_q"}
                rec.update(step=step, total=total, lr=opt.param_groups[0]["lr"], gnorm=float(gn),
                           tok_per_s=tok_acc / dt, s_per_step=dt / (n_micro / cfg.grad_accum),
                           q_per_s=acc["n_q"] / dt, mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                           elapsed_min=(time.time() - t_train) / 60)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                acc, tok_acc, t_step = {}, 0, time.time()
            if step % cfg.save_every == 0:
                save(f"{cfg.out_dir}/ckpt_last.pt", step)
            if step >= total:
                done = True
                break
    save(f"{cfg.out_dir}/ckpt_final.pt", step)
    print(f"[done] {step} steps in {(time.time()-t_train)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
