"""AR presence-only read-out in ONE padded block pass (supplements ablation/latency_v2.py, whose presence-only
timing starts the question blocks in equal-length groups because the incremental decoder needs exact GDN states).
Reading q(t1 = none) needs only the logits at <box_start>, so all N question blocks can run in a single right-padded
pass (net.block, as in eval_ab.ar_decode step 0), exactly like ours. Same workload / modes as latency_v2.

python -m ablation.latency_presence --ar CKPT_AR --out OUT.json [--merge_lora]
"""
import argparse, json, time
import numpy as np
import torch

from ablation.common import ABConfig, ARDataset, collate_ar, build_ar, ar_load
from ablation.train_ab import to_dev
from ablation.latency_v2 import MODES, merge_lora, stats, items, n_calls_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", required=True)
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = torch.load(a.ar, map_location="cpu", weights_only=False)
    cfg = ABConfig().update({k: v for k, v in sd["cfg"].items() if k in ABConfig.__dataclass_fields__})
    net, fmt, proc = build_ar(cfg, "cuda")
    ar_load(a.ar, net)
    res = dict(gpu=torch.cuda.get_device_name(), n_images=a.n_images, merge_lora=a.merge_lora)
    if a.merge_lora:
        res["merged_ar"] = merge_lora(net.hf)
    net.eval()
    ds = ARDataset(f"{a.prep_dir}/eval_refcoco_val.jsonl", cfg.image_root, proc, cfg, train=False, fmt=fmt)
    res["ar_digits_presence_1pass"] = {}
    for ipb, nq in MODES:
        T = []
        for c in range(n_calls_for(ipb, nq, a.n_images) + a.warmup):
            idx = [(c * ipb + j) % len(ds) for j in range(ipb)]
            batch = collate_ar(items(ds, idx, nq), ds.tok, with_answer=False); batch.pop("meta")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                b = to_dev(batch, "cuda")
                feats = net.image_features(b["pixel_values"], b["grid_thw"])
                pc = net.prefill(b, feats)
                hq = net.block(pc, b["q_ids"], b["q_valid"], b["q_pos"], b["q_img"], b["q_last"])
                last = b["q_valid"].sum(1) - 1
                lg = net.logits(hq[torch.arange(hq.shape[0], device=hq.device), last]).float()
                _ = lg.softmax(-1)[:, fmt.none].cpu()
            torch.cuda.synchronize()
            if c >= a.warmup:
                T.append((time.perf_counter() - t0) * 1000)
        name = f"bs{ipb}_N{nq}"
        res["ar_digits_presence_1pass"][name] = stats(T, T, ipb * nq)
        print("ar_presence_1pass", name, json.dumps(res["ar_digits_presence_1pass"][name]), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
