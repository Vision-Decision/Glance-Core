"""Leak test: readout at <box_start> must be identical when two different equal-length answers are appended."""
import torch
from ablation.common import ABConfig, ARDataset, collate_ar, build_ar
from ablation.train_ab import to_dev
cfg = ABConfig().update(dict(variant="ar_atomic", grad_ckpt=False))
net, fmt, proc = build_ar(cfg, "cuda"); net.eval()
ds = ARDataset(cfg.train_file, cfg.image_root, proc, cfg, limit=4, fmt=fmt, train=False)
items = [ds[0], ds[1]]
def run(items):
    b = to_dev({k: v for k, v in collate_ar(items, ds.tok).items() if k != "meta"}, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        feats = net.image_features(b["pixel_values"], b["grid_thw"])
        _, hq = net(b, feats)
    return hq[torch.arange(hq.shape[0]), b["q_last"]].float()
a = run(items)
items2 = [dict(it, a_ids=[[fmt.V0 + 7] * len(x) for x in it["a_ids"]]) for it in items]
c = run(items2)
print("diff readout, different answers same length:", (a - c).abs().max().item())
items3 = [dict(it, a_ids=[x + [fmt.box_end] * 3 for x in it["a_ids"]]) for it in items]
print("diff readout, longer answers:", (a - run(items3)).abs().max().item())
