"""Fair latency (experiment 5 b/c): Glance (ours) vs the AR-digits head with incremental decoding, same engine.

python -m ablation.latency_v2 --ours CKPT_OURS --ar CKPT_AR --out OUT.json [--merge_lora]

Workload (identical in ablation/latency_vllm.py): image i = i-th record of eval_refcoco_val.jsonl, resized to
<= 640 visual tokens (sides multiples of 32), N questions = the record's referring expressions cycled to N.
Modes: bs1_N{1,2,4,8,16,32} (one image, N questions per call) and bs8_N{1,8} (8 images per call, throughput).
Per call we time host->device copy, vision encoder, prefill, readout/decoding and device->host of all answers
(image decode/resize excluded, identical for all systems).
  ours                    one prefill + one block forward + heads
  ar_digits_inc           prefill + question blocks + grammar-constrained greedy with a per-token KV cache
                          (attention) / recurrent-state cache (GDN) over the shared prefix (ablation/incdec.py)
  ar_digits_presence      AR presence-only read-out: prefill + question blocks, read q(t1 = none); no decoding
Metrics: ms median / p90 per call, questions/s, time-to-first-answer (TTFA; ours and presence-only: = call time;
AR: until the first question's answer is complete), mean generated tokens.
LoRA is merged into the base weights for both models (--merge_lora).
"""
import argparse, json, time
import numpy as np
import torch

from ablation.common import ABConfig, ARDataset, collate_ar, build_ar, ar_load
from ablation.dvcore.dataset import RecordDataset, collate
from ablation.train_ab import run_ours, to_dev
from ablation import incdec

MODES = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (1, 32), (8, 1), (8, 8)]


def merge_lora(hf):
    n = 0
    for m in hf.modules():
        if hasattr(m, "lora_A") and hasattr(m, "merge"):
            m.merge(); n += 1
    return n


def stats(times, ttfa, nq, extra=None):
    t, f = np.array(times), np.array(ttfa)
    d = dict(ms_median=float(np.median(t)), ms_p90=float(np.percentile(t, 90)), ms_mean=float(t.mean()),
             q_per_s=float(nq * len(t) / (t.sum() / 1000)), ttfa_ms_median=float(np.median(f)),
             ttfa_ms_p90=float(np.percentile(f, 90)), n_calls=len(t), questions_per_call=nq)
    d.update(extra or {})
    return d


def items(ds, idx, nq):
    out = []
    for i in idx:
        it = ds[i]
        n0 = len(it["q_ids"])
        cyc = lambda L: [L[j % n0] for j in range(nq)]
        it = dict(it, q_ids=cyc(it["q_ids"]), present=[True] * nq, box=[[0, 0, 1, 1]] * nq, qs=cyc(it["qs"]))
        it["a_ids"] = [[]] * nq
        out.append(it)
    return out


def n_calls_for(ipb, nq, n_images):
    n = n_images // ipb
    return n if nq <= 8 else max(20, n // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, n_images=a.n_images, merge_lora=a.merge_lora)
    path = f"{a.prep_dir}/eval_refcoco_val.jsonl"

    # ---------------- ours
    sd = torch.load(a.ours, map_location="cpu", weights_only=False)
    cfg = ABConfig().update({k: v for k, v in sd["cfg"].items() if k in ABConfig.__dataclass_fields__})
    from ablation.dvcore.build import build, load_ckpt
    net, heads, proc = build(cfg, "cuda")
    load_ckpt(a.ours, net, heads)
    if a.merge_lora:
        res["merged_ours"] = merge_lora(net.hf)
    net.eval(); heads.eval()
    ds = RecordDataset(path, cfg.image_root, proc, cfg, train=False)
    tok = ds.tok
    res["ours"] = {}
    for ipb, nq in MODES:
        times = []
        vis = []
        for c in range(n_calls_for(ipb, nq, a.n_images) + a.warmup):
            idx = [(c * ipb + j) % len(ds) for j in range(ipb)]
            batch = collate(items(ds, idx, nq), tok); batch.pop("meta")
            vis.append(int(batch["vis_valid"].sum()) / ipb)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                out, _ = run_ours(net, heads, to_dev(batch, "cuda"), train=False)
                _ = (out["presence_logit"].float().sigmoid().cpu(), out["box_pred"].float().cpu(), out["iou_logit"].float().cpu())
            torch.cuda.synchronize()
            if c >= a.warmup:
                times.append((time.perf_counter() - t0) * 1000)
        name = f"bs{ipb}_N{nq}"
        res["ours"][name] = stats(times, times, ipb * nq, dict(vis_tokens_mean=float(np.mean(vis))))
        print("ours", name, json.dumps(res["ours"][name]), flush=True)
    del net, heads
    torch.cuda.empty_cache()

    # ---------------- AR digits
    sd = torch.load(a.ar, map_location="cpu", weights_only=False)
    cfg = ABConfig().update({k: v for k, v in sd["cfg"].items() if k in ABConfig.__dataclass_fields__})
    net, fmt, proc = build_ar(cfg, "cuda")
    ar_load(a.ar, net)
    if a.merge_lora:
        res["merged_ar"] = merge_lora(net.hf)
    net.eval()
    ds = ARDataset(path, cfg.image_root, proc, cfg, train=False, fmt=fmt)
    tok = ds.tok
    res["ar_digits_inc"], res["ar_digits_presence"] = {}, {}
    for ipb, nq in MODES:
        T, F, TP, ntok, nnone = [], [], [], [], []
        for c in range(n_calls_for(ipb, nq, a.n_images) + a.warmup):
            idx = [(c * ipb + j) % len(ds) for j in range(ipb)]
            batch = collate_ar(items(ds, idx, nq), tok, with_answer=False); batch.pop("meta")
            for full in (False, True):
                first = []
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    b = to_dev(batch, "cuda")
                    feats = net.image_features(b["pixel_values"], b["grid_thw"])
                    pc = net.prefill(b, feats)
                    cache, lg0 = incdec.start_blocks(net, pc, b)
                    if full:
                        g = incdec.greedy(net, fmt, cache, lg0, force_box=False,
                                          on_first_done=lambda: first.append(time.perf_counter()))
                    else:
                        _ = lg0.softmax(-1)[:, fmt.none].cpu()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                if c >= a.warmup:
                    if full:
                        T.append((t1 - t0) * 1000); F.append(((first[0] if first else t1) - t0) * 1000)
                        ntok.append(float(np.mean([len(x[0]) for x in g])))
                        nnone.append(float(np.mean([x[0][:1] == [fmt.none] for x in g])))
                    else:
                        TP.append((t1 - t0) * 1000)
        name = f"bs{ipb}_N{nq}"
        res["ar_digits_inc"][name] = stats(T, F, ipb * nq, dict(gen_tokens_mean=float(np.mean(ntok)),
                                                                none_rate=float(np.mean(nnone))))
        res["ar_digits_presence"][name] = stats(TP, TP, ipb * nq)
        print("ar_inc", name, json.dumps(res["ar_digits_inc"][name]), flush=True)
        print("ar_presence", name, json.dumps(res["ar_digits_presence"][name]), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
