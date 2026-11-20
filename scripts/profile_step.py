"""Profile a few training steps: python scripts/profile_step.py [--set ...]"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.profiler import profile, ProfilerActivity
from dv.config import Config
from dv.data.dataset import RecordDataset, collate
from dv.model.build import build
from dv.model.heads import box_loss
from train import run_model, to_dev
cfg, _ = Config.parse()
net, heads, proc = build(cfg); net.train(); heads.train()
ds = RecordDataset(cfg.train_file, cfg.image_root, proc, cfg, limit=200)
dl = torch.utils.data.DataLoader(ds, batch_size=cfg.images_per_step, num_workers=4, collate_fn=lambda it: collate(it, ds.tok))
it = iter(dl)
def step():
    b = to_dev(next(it), "cuda")
    out, ct = run_model(net, heads, b)
    loss, _ = box_loss(out, dict(present=b["present"], box=b["box"], center_t=ct), cfg)
    loss.backward()
for _ in range(3): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    t = time.time()
    for _ in range(2): step()
    torch.cuda.synchronize(); print("2 steps", time.time() - t)
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12, max_name_column_width=60))
rows = [e for e in prof.key_averages(group_by_input_shape=True) if e.key in ("aten::fill_", "aten::zero_", "aten::zeros", "aten::copy_")]
rows.sort(key=lambda e: -e.self_device_time_total)
for e in rows[:12]:
    print(e.key, e.count, round(e.self_device_time_total / 1e3, 1), "ms", str(e.input_shapes)[:150])
rows = [e for e in prof.key_averages(group_by_stack_n=12) if e.key == "aten::fill_"]
rows.sort(key=lambda e: -e.self_device_time_total)
for e in rows[:4]:
    print("== fill_", e.count, round(e.self_device_time_total / 1e3, 1), "ms"); print("\n".join(e.stack[:12]))
