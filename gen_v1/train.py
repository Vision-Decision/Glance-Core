"""Train Gen-v1 (generative baseline of Glance v1).  python -m gen_v1.train --cfg configs/v1_4b.json --set out_dir=...

Same config file, optimizer, schedule, LoRA, data order (TokenBudgetSampler seed), steps and checkpointing as train.py
(task v1); only the outputs and the loss differ (gen_v1/common.py). There are no heads: one parameter group (LoRA, lr).
"""
import json, math, os, random, time
import torch

from dv.config import Config
from dv.data.mixed import TokenBudgetSampler
from gen_v1.common import GenDataset, collate_gen, build_gen, gen_state, gen_load, gen_loss
from train import to_dev, _skip


def save(path, net, cfg, step, opt=None):
    tmp = path + ".tmp"
    torch.save(dict(**gen_state(net), cfg=cfg.__dict__, extra=dict(step=step)), tmp)
    os.replace(tmp, path)
    if opt is not None:
        op = os.path.join(os.path.dirname(path), "opt_last.pt")
        torch.save(dict(step=step, opt=opt.state_dict()), op + ".tmp")
        os.replace(op + ".tmp", op)


def main():
    cfg, _ = Config.parse()
    assert cfg.task == "v1"
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    open(f"{cfg.out_dir}/config.json", "w").write(cfg.to_json())
    dev = "cuda"
    t0 = time.time()
    net, fmt, proc = build_gen(cfg, dev)
    net.train()
    print(f"[load] {time.time()-t0:.1f}s", flush=True)
    ds = GenDataset(cfg.train_file, cfg.data_root, proc, cfg, limit=cfg.limit_records, fmt=fmt)
    sampler = TokenBudgetSampler(ds, cfg.prefix_budget, cfg.q_budget, cfg.max_records, seed=cfg.seed)
    tok = ds.tok
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=cfg.num_workers,
                                     collate_fn=lambda it: collate_gen(it, tok),
                                     persistent_workers=False, pin_memory=True, prefetch_factor=4 if cfg.num_workers else None)
    steps_per_epoch = math.ceil(len(dl) / cfg.grad_accum)
    total = cfg.max_steps or int(steps_per_epoch * cfg.epochs)
    lora_p = [p for n, p in net.hf.named_parameters() if p.requires_grad]
    print(f"[params] lora={sum(p.numel() for p in lora_p)/1e6:.1f}M records={len(ds)} micro_batches={len(dl)} steps={total}",
          flush=True)
    opt = torch.optim.AdamW([dict(params=lora_p, lr=cfg.lr)], weight_decay=cfg.weight_decay, betas=(0.9, 0.95), fused=True)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_factor(s):
        if s < cfg.warmup_steps:
            return (s + 1) / cfg.warmup_steps
        p = (s - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    step = 0
    if cfg.resume:
        sd = gen_load(cfg.resume, net)
        step = sd["extra"].get("step", 0)
        op = os.path.join(os.path.dirname(cfg.resume), "opt_last.pt")
        if os.path.exists(op):
            o = torch.load(op, map_location="cpu", weights_only=False)
            if o.get("step") == step:
                opt.load_state_dict(o["opt"]); print(f"[resume] optimizer state from {op}", flush=True)
        print(f"[resume] from {cfg.resume} step {step}", flush=True)
    logf = open(f"{cfg.out_dir}/train_log.jsonl", "a")
    torch.cuda.reset_peak_memory_stats()
    micro, acc, tok_acc, t_step, wait_acc = 0, {}, 0, time.time(), 0.0
    t_train = t_save = time.time()
    done, epoch = False, 0
    skip = step * cfg.grad_accum if cfg.resume else 0
    if skip >= len(dl):
        epoch, skip = skip // len(dl), skip % len(dl)
        sampler.set_epoch(epoch)
    while not done:
        t_w = time.time()
        for bidx, batch in enumerate(dl) if not skip else _skip(dl, skip):
            skip = 0
            wait_acc += time.time() - t_w
            batch.pop("meta", None)
            batch = to_dev(batch, dev)
            loss, parts = gen_loss(net, batch, cfg)
            if not torch.is_tensor(loss):
                t_w = time.time(); continue
            (loss / cfg.grad_accum).backward()
            micro += 1
            tok_acc += batch["n_tokens"]
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
                acc["_cnt_" + k] = acc.get("_cnt_" + k, 0) + 1
            acc["loss"] = acc.get("loss", 0.0) + float(loss)
            acc["n_q"] = acc.get("n_q", 0) + len(batch["present"])
            if micro % cfg.grad_accum:
                t_w = time.time(); continue
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_factor(step)
            gn = torch.nn.utils.clip_grad_norm_(lora_p, cfg.clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                torch.cuda.synchronize()
                dt = time.time() - t_step
                n_micro = cfg.grad_accum * (cfg.log_every if step > 1 else 1)
                rec = {k: v / acc.get("_cnt_" + k, n_micro) for k, v in acc.items() if k != "n_q" and not k.startswith("_cnt_")}
                rec.update(step=step, total=total, lr=opt.param_groups[0]["lr"], gnorm=float(gn), tok_per_s=tok_acc / dt,
                           s_per_step=dt / (n_micro / cfg.grad_accum), q_per_s=acc["n_q"] / dt,
                           mem_gb=torch.cuda.max_memory_allocated() / 2**30, data_wait_frac=wait_acc / dt,
                           elapsed_min=(time.time() - t_train) / 60)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                acc, tok_acc, t_step, wait_acc = {}, 0, time.time(), 0.0
            if cfg.save_every_min > 0 and time.time() - t_save > cfg.save_every_min * 60:
                save(f"{cfg.out_dir}/ckpt_last.pt", net, cfg, step, opt)
                print(f"[save] step {step}", flush=True)
                t_save = time.time()
            if step >= total:
                done = True
                break
            t_w = time.time()
        if not done:
            epoch += 1
            sampler.set_epoch(epoch)
    save(f"{cfg.out_dir}/ckpt_final.pt", net, cfg, step)
    print(f"[done] {step} steps in {(time.time()-t_train)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
