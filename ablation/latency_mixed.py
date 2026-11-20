"""Mixed typed prefill latency (round-3 request): Glance v1 (dv/ model, box + yes/no + choice + score heads) answers
Q mixed questions about one image in one prefill + one block pass. Workload: ablation/mixed_q.py (same as
latency_vllm.py --mixed). Q in {1,2,4,8,16,32}, one image per call, 100 images (50 for Q>=16), 5 warm-up calls.
Timed: host->device, vision encoder, prefill, question blocks, all heads, device->host (preprocessing excluded).

python -m ablation.latency_mixed --ckpt CKPT_V1 --out OUT.json [--data_root /content/data] [--merge_lora]
"""
import argparse, json, time
import numpy as np
import torch

from dv.config import Config
from dv.model.build import build, load_ckpt
from dv.data.mixed import MixedDataset, collate_mixed
from train import run_model_v1, to_dev
from ablation.mixed_q import mixed_questions
from ablation.latency_v2 import merge_lora, stats

QS = [1, 2, 4, 8, 16, 32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--data_root", default="/content/data")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update(sd["cfg"])
    net, heads, proc = build(cfg, "cuda")
    load_ckpt(a.ckpt, net, heads)
    res = dict(gpu=torch.cuda.get_device_name(), ckpt=a.ckpt, n_images=a.n_images)
    if a.merge_lora:
        res["merged"] = merge_lora(net.hf)
    net.eval(); heads.eval()
    src = [json.loads(l) for l in open(a.prep)][: a.n_images + a.warmup]
    res["modes"] = {}
    for Q in QS:
        recs = [dict(src="eval", img=f"coco/train2014/{r['file']}", wh=[r["W"], r["H"]], max_vis=640, plen=0,
                     questions=[dict(q, present=True, box=None) if q["type"] == "box" else q
                                for q in mixed_questions([q["text"] for q in r["questions"]], Q)]) for r in src]
        ds = MixedDataset(None, a.data_root, proc, cfg, train=False, recs=recs)
        n = a.n_images if Q <= 8 else a.n_images // 2
        T, nvis = [], []
        for i in range(n + a.warmup):
            batch = collate_mixed([ds[i % len(ds)]], ds.tok); batch.pop("meta", None)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                r = run_model_v1(net, heads, to_dev(batch, "cuda"), train=False)
                _ = [v.float().cpu() for v in ([r["typed_logits"]] if "typed_logits" in r else []) +
                     ([v for v in r["box"].values() if torch.is_tensor(v)] if "box" in r else [])]
            torch.cuda.synchronize()
            if i >= a.warmup:
                T.append((time.perf_counter() - t0) * 1000)
        res["modes"][f"bs1_Q{Q}"] = stats(T, T, Q, dict(types=[q["type"] for q in recs[0]["questions"]]))
        print(f"bs1_Q{Q}", json.dumps(res["modes"][f"bs1_Q{Q}"]), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
