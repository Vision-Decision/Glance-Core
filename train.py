"""Train the v0 box model or the v1 typed decision model (cfg.task).
python train.py --cfg configs/v0_4b.json [--set key=value ...]      python train.py --cfg configs/v1_4b.json"""
import json, math, os, random, time
import torch

from dv.config import Config
from dv.data.dataset import RecordDataset, collate
from dv.model.build import build, save_ckpt, load_ckpt
from dv.model.heads import center_target, box_loss, center_target_box, center_target_points, box_loss_v1
from dv.model.typed import typed_logits, typed_loss
from dv.model.v1forward import run_model_v1  # noqa: F401  (re-exported for eval.py / scripts)


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_model(net, heads, batch, train=True):
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
        # condition edges on a patch sampled from the soft center target (robust to off-by-one center choice);
        # absent questions: any valid patch (their box losses are masked anyway)
        sel = torch.multinomial(ct + (~batch["present"])[:, None] * batch["vis_valid"][q_img].float() + 1e-12, 1)[:, 0]
    out = heads(hq_last, vis, batch["vis_valid"], batch["centers"], q_img, sel_idx=sel)
    return out, ct


def loss_v1(res, batch, cfg):
    total, parts = 0.0, {}
    if "box" in res:
        bi = res["box_idx"]
        bt = dict(present=batch["present"][bi], has_box=batch["has_box"][bi], box=res.get("box_t", batch["box"][bi]),
                  center_t=res["center_t"])
        if "gt" in res:
            bt.update(gt=res["gt"], gmask=res["gmask"])
        bl, bp = box_loss_v1(res["box"], bt, cfg)
        if "multi_t" in res:
            from dv.model.multibox import multi_loss
            ml, mp = multi_loss(res["box"]["multi"], res["multi_t"], cfg)
            bl = (bl.detach() if cfg.train_only == "multi" else bl) + ml
            bp.update(mp); bp["multi_total"] = ml.detach()
        total = total + cfg.w_box * bl
        parts.update(bp); parts["box_total"] = bl.detach(); parts["n_box"] = float(len(bi))
        if cfg.w_cons > 0 and "grp" in batch:
            cl, cp = consistency_loss(res["box"], batch["grp"][bi], batch["anchor"][bi], batch["present"][bi], cfg)
            if cl is not None:
                total = total + cfg.w_cons * cl
                parts.update(cp)
    if "typed_logits" in res:
        m = res["typed_mask"]
        tl, tp = typed_loss(res["typed_logits"], batch["tgt"][m], batch["lab_idx"][m] >= 0, batch["qtype"][m], cfg)
        total = total + cfg.w_typed * (tl.detach() if cfg.train_only == "multi" else tl)
        parts.update(tp); parts["typed_total"] = tl.detach()
    return total, parts


def consistency_loss(out, grp, anchor, present, cfg):
    """Self-distillation from the original (anchor) description of a target to its shifted descriptions (zh /
    paraphrase / functional / vague) in the same image: presence, patch distribution and decoded box of each member are
    pulled toward the anchor's (stop-gradient). Returns (loss, parts) or (None, {}) if the batch has no pairs."""
    import torch.nn.functional as F
    g = grp.tolist(); an = anchor.tolist()
    first = {}
    for i, (gi, a) in enumerate(zip(g, an)):
        if gi >= 0 and a and gi not in first:
            first[gi] = i
    pr = present.tolist()
    # present groups only (S3 sprint: pulling absent members toward their anchor cost ~5 pts N-acc)
    mem = [(i, first[gi]) for i, (gi, a) in enumerate(zip(g, an))
           if gi >= 0 and not a and gi in first and (pr[i] and pr[first[gi]] or not cfg.cons_present_only)]
    if not mem:
        return None, {}
    dev = grp.device
    i = torch.tensor([m[0] for m in mem], device=dev); j = torch.tensor([m[1] for m in mem], device=dev)
    pl = out["presence_logit"]
    lp = F.binary_cross_entropy_with_logits(pl[i], pl[j].detach().sigmoid())
    loss = cfg.w_cons_pres * lp
    parts = dict(cons_pres=lp.detach(), n_cons=float(len(mem)))
    pm = present[i]
    if pm.any():
        ii, jj = i[pm], j[pm]
        la = out["center_logits"][jj].detach().log_softmax(-1)
        lm = out["center_logits"][ii].log_softmax(-1)
        kl = (la.exp() * (la - lm)).sum(-1).mean()
        lb = (out["box_pred"][ii] - out["box_pred"][jj].detach()).abs().sum(-1).mean()
        loss = loss + cfg.w_cons_center * kl + cfg.w_cons_box * lb
        parts.update(cons_center=kl.detach(), cons_box=lb.detach())
    return loss, parts


class CaptionLM:
    """Small next-token loss on image captions (EN + ZH) through the plain HF forward of the same LoRA'd model, to keep
    the base model's language ability. Records: {"img": path relative to data_root, "prompt": str, "caption": str}."""

    def __init__(self, cfg, proc):
        from PIL import Image  # noqa: F401
        self.cfg, self.proc = cfg, proc
        self.recs = [json.loads(l) for l in open(cfg.lm_file)]
        self.rng = random.Random(cfg.seed + 7)
        self.i = len(self.recs)

    def _next(self):
        if self.i >= len(self.recs):
            self.rng.shuffle(self.recs); self.i = 0
        self.i += 1
        return self.recs[self.i - 1]

    def loss(self, net):
        from PIL import Image
        from dv.data.dataset import resize_dims
        tot, n = 0.0, 0
        for _ in range(self.cfg.lm_bs):
            r = self._next()
            img = Image.open(f"{self.cfg.data_root}/{r['img']}").convert("RGB")
            w, h = resize_dims(*img.size, getattr(self.cfg, "lm_max_vis", 448), 2.0)
            img = img.resize((w, h), Image.BICUBIC)
            msgs = ([{"role": "system", "content": [{"type": "text", "text": r["system"]}]}] if r.get("system") else []) + \
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["prompt"]}]}]
            prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False) + r.get("prefill", "")
            full = prompt + r["caption"] + "<|im_end|>"
            inp = self.proc(text=[full], images=[img], return_tensors="pt", images_kwargs=dict(do_resize=False)).to("cuda")
            n_prompt = self.proc(text=[prompt], images=[img], return_tensors="pt", images_kwargs=dict(do_resize=False))["input_ids"].shape[1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net.hf(**inp, use_cache=False).logits[0, n_prompt - 1:-1].float()
            tgt = inp["input_ids"][0, n_prompt:]
            l = torch.nn.functional.cross_entropy(logits, tgt)
            (l * self.cfg.w_lm / self.cfg.lm_bs / self.cfg.grad_accum).backward()
            tot += float(l); n += 1
        return tot / max(1, n)


def make_loader_v1(cfg, proc, epoch=0):
    from dv.data.mixed import MixedDataset, collate_mixed, TokenBudgetSampler
    ds = MixedDataset(cfg.train_file, cfg.data_root, proc, cfg, limit=cfg.limit_records)
    sampler = TokenBudgetSampler(ds, cfg.prefix_budget, cfg.q_budget, cfg.max_records, seed=cfg.seed)
    tok = ds.tok
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=cfg.num_workers,
                                     collate_fn=lambda it: collate_mixed(it, tok),
                                     persistent_workers=False, pin_memory=True, prefetch_factor=4 if cfg.num_workers else None)
    return ds, sampler, dl


def main():
    cfg, _ = Config.parse()
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    open(f"{cfg.out_dir}/config.json", "w").write(cfg.to_json())
    dev = "cuda"
    t0 = time.time()
    net, heads, proc = build(cfg, dev)
    net.train(); heads.train()
    print(f"[load] {time.time()-t0:.1f}s; layer_types={net.layer_types.count('full_attention')} full / "
          f"{net.layer_types.count('linear_attention')} linear; hidden={net.hidden}", flush=True)
    v1 = cfg.task == "v1"
    sampler = None
    lm = CaptionLM(cfg, proc) if (v1 and cfg.w_lm > 0 and cfg.lm_file) else None
    if v1:
        ds, sampler, dl = make_loader_v1(cfg, proc)
    else:
        ds = RecordDataset(cfg.train_file, cfg.image_root, proc, cfg, limit=cfg.limit_records)
        tok = ds.tok
        dl = torch.utils.data.DataLoader(ds, batch_size=cfg.images_per_step, shuffle=True, drop_last=False,
                                         num_workers=cfg.num_workers, collate_fn=lambda it: collate(it, tok),
                                         persistent_workers=cfg.num_workers > 0, pin_memory=True)
    steps_per_epoch = math.ceil(len(dl) / cfg.grad_accum)
    total = cfg.max_steps or int(steps_per_epoch * cfg.epochs)
    if cfg.train_only == "multi":  # v1.2 add-on: LoRA and every pre-existing head stay bit-identical to init_ckpt
        assert cfg.multibox and (cfg.init_ckpt or cfg.resume), "train_only=multi needs multibox and a checkpoint"
        for n, p in list(net.hf.named_parameters()) + list(heads.named_parameters()):
            if not n.startswith("box.multi."):
                p.requires_grad_(False)
    elif cfg.train_only:
        raise ValueError(cfg.train_only)
    lora_p = [p for n, p in net.hf.named_parameters() if p.requires_grad]
    head_p = [p for n, p in heads.named_parameters() if p.requires_grad and not n.startswith("box.multi.")]
    multi_p = [p for n, p in heads.named_parameters() if p.requires_grad and n.startswith("box.multi.")]
    # v1.2b fix: the heat map (box.multi.h_*) can have its own (lower) LR and its own grad clip
    heat_p = [p for n, p in heads.named_parameters() if p.requires_grad and n.startswith("box.multi.h_")] if cfg.lr_heat else []
    if heat_p:
        multi_p = [p for p in multi_p if not any(p is q for q in heat_p)]
    n_lora = sum(p.numel() for p in lora_p); n_head = sum(p.numel() for p in head_p + multi_p)
    print(f"[params] lora={n_lora/1e6:.1f}M heads={n_head/1e6:.1f}M records={len(ds)} steps={total}", flush=True)
    groups = [g for g in (dict(params=lora_p, lr=cfg.lr), dict(params=head_p, lr=cfg.lr_heads),
                          dict(params=multi_p, lr=cfg.lr_multi or cfg.lr_heads), dict(params=heat_p, lr=cfg.lr_heat)) if g["params"]]
    head_p = head_p + multi_p + heat_p
    if cfg.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95), fused=True)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_factor(s):
        if s < cfg.warmup_steps:
            return (s + 1) / cfg.warmup_steps
        p = (s - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    step = 0
    if cfg.init_ckpt and not cfg.resume:  # Glance-Video: initialise from v1 / v1.1 (LoRA + heads)
        load_ckpt(cfg.init_ckpt, net, heads)
        print(f"[init] LoRA + heads from {cfg.init_ckpt}", flush=True)
    if cfg.resume:
        sd = load_ckpt(cfg.resume, net, heads)
        step = sd["extra"].get("step", 0)
        if "opt" in sd["extra"]:
            opt.load_state_dict(sd["extra"]["opt"])
        opt_path = os.path.join(os.path.dirname(cfg.resume), "opt_last.pt")
        if os.path.exists(opt_path):
            o = torch.load(opt_path, map_location="cpu", weights_only=False)
            if o.get("step") == step:
                opt.load_state_dict(o["opt"]); print(f"[resume] optimizer state from {opt_path}", flush=True)
        print(f"[resume] from {cfg.resume} step {step}", flush=True)
    logf = open(f"{cfg.out_dir}/train_log.jsonl", "a")
    torch.cuda.reset_peak_memory_stats()
    micro, acc, tok_acc, t_step, wait_acc = 0, {}, 0, time.time(), 0.0
    t_train = time.time()
    t_save = time.time()
    done = False
    epoch = 0
    skip = step * cfg.grad_accum if (cfg.resume and v1) else 0   # v1 resume: skip micro-batches already seen
    if v1 and skip >= len(dl):
        epoch, skip = skip // len(dl), skip % len(dl)
        sampler.set_epoch(epoch)
    while not done:
        t_w = time.time()
        for bidx, batch in enumerate(dl) if not skip else _skip(dl, skip):
            skip = 0
            wait_acc += time.time() - t_w
            batch = to_dev(batch, dev)
            if v1:
                res = run_model_v1(net, heads, batch, train=True)
                loss, parts = loss_v1(res, batch, cfg)
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    t_w = time.time(); continue
            else:
                out, ct = run_model(net, heads, batch, train=True)
                loss, parts = box_loss(out, dict(present=batch["present"], box=batch["box"], center_t=ct), cfg)
            (loss / cfg.grad_accum).backward()
            if lm is not None:
                parts["lm"] = lm.loss(net)
            micro += 1
            tok_acc += batch["n_tokens"]
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
                acc["_cnt_" + k] = acc.get("_cnt_" + k, 0) + 1
            acc["loss"] = acc.get("loss", 0.0) + float(loss)
            acc["n_q"] = acc.get("n_q", 0) + len(batch["present"])
            if micro % cfg.grad_accum:
                t_w = time.time()
                continue
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_factor(step)
            if heat_p:
                acc["_gn_heat"] = acc.get("_gn_heat", 0.0) + float(torch.nn.utils.clip_grad_norm_(heat_p, cfg.clip_heat or cfg.clip))
                acc["_cnt__gn_heat"] = acc.get("_cnt__gn_heat", 0) + 1
            if multi_p and cfg.clip_multi_separately:  # a heat-loss spike must not set the direction of the clipped step
                gnm = torch.nn.utils.clip_grad_norm_(multi_p, cfg.clip)
                rest = lora_p + [p for p in head_p if not any(p is q for q in multi_p) and not any(p is q for q in heat_p)]
                gn = torch.nn.utils.clip_grad_norm_(rest, cfg.clip) if rest else gnm
            else:
                gn = torch.nn.utils.clip_grad_norm_(lora_p + head_p, cfg.clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                torch.cuda.synchronize()
                dt = time.time() - t_step
                n_micro = cfg.grad_accum * (cfg.log_every if step > 1 else 1)
                rec = {k: v / acc.get("_cnt_" + k, n_micro) for k, v in acc.items()
                       if k != "n_q" and not k.startswith("_cnt_")}
                rec.update(step=step, total=total, lr=opt.param_groups[0]["lr"], gnorm=float(gn),
                           tok_per_s=tok_acc / dt, s_per_step=dt / (n_micro / cfg.grad_accum),
                           q_per_s=acc["n_q"] / dt, mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                           data_wait_frac=wait_acc / dt, elapsed_min=(time.time() - t_train) / 60)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                acc, tok_acc, t_step, wait_acc = {}, 0, time.time(), 0.0
            timed = cfg.save_every_min > 0 and time.time() - t_save > cfg.save_every_min * 60
            if step % cfg.save_every == 0 or timed:
                save_ckpt(f"{cfg.out_dir}/ckpt_last.tmp", net, heads, cfg, dict(step=step))
                os.replace(f"{cfg.out_dir}/ckpt_last.tmp", f"{cfg.out_dir}/ckpt_last.pt")
                if cfg.save_every_min > 0:
                    torch.save(dict(step=step, opt=opt.state_dict()), f"{cfg.out_dir}/opt_last.tmp")
                    os.replace(f"{cfg.out_dir}/opt_last.tmp", f"{cfg.out_dir}/opt_last.pt")
                print(f"[save] step {step}", flush=True)
                t_save = time.time()
            if step >= total:
                done = True
                break
            t_w = time.time()
        if v1 and not done:
            epoch += 1
            sampler.set_epoch(epoch)
    save_ckpt(f"{cfg.out_dir}/ckpt_final.pt", net, heads, cfg, dict(step=step))
    print(f"[done] {step} steps in {(time.time()-t_train)/60:.1f} min", flush=True)


def _skip(dl, n):
    """Resume helper: iterate a loader from micro-batch n (batch order is deterministic per epoch)."""
    bs = dl.batch_sampler
    full = bs._batches
    bs._batches = full[n:]
    try:
        for i, b in enumerate(dl):
            yield n + i, b
    finally:
        bs._batches = full


if __name__ == "__main__":
    main()
