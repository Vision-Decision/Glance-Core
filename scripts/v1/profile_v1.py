"""Per-batch timing + torch.profiler for the v1 mixed loader.  python scripts/v1/profile_v1.py [--set k=v ...]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from dv.config import Config
from dv.model.build import build
from train import make_loader_v1, run_model_v1, loss_v1, to_dev

cfg, _ = Config.parse(["--cfg", "configs/v1_4b.json"] + sys.argv[1:])
net, heads, proc = build(cfg, "cuda")
net.train(); heads.train()
ds, sampler, dl = make_loader_v1(cfg, proc)
it = iter(dl)
for i in range(24):
    b = next(it); b.pop("meta")
    kind = "img" if b["pixel_values"] is not None else "txt"
    B, Lp = b["p_ids"].shape; M, Tq = b["q_ids"].shape
    b = to_dev(b, "cuda")
    torch.cuda.synchronize(); t0 = time.time()
    res = run_model_v1(net, heads, b); loss, _ = loss_v1(res, b, cfg); loss.backward()
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"{i:2d} {kind} B={B} Lp={Lp} M={M} Tq={Tq} prefix_tok={B*Lp} q_tok={M*Tq} valid={b['n_tokens']} {dt:.2f}s "
          f"-> {b['n_tokens']/dt:.0f} tok/s", flush=True)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for i in range(4):
        b = next(it); b.pop("meta"); b = to_dev(b, "cuda")
        res = run_model_v1(net, heads, b); loss, _ = loss_v1(res, b, cfg); loss.backward()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
