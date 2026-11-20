"""Dump per-question box outputs + analytic box-quality features (arch/quality/edgeq.py) for one checkpoint.

  ablation ckpt (controlled study, v0 prompt):  python -m arch.quality.dump --kind ab --ckpt ours_neg25_bf16.pt \
        --sets grefcoco_val_merged,arcal_gtest --out DIR
  v1-family ckpt:                              python -m arch.quality.dump --kind v1 --ckpt v1.1/ckpt_final.pt \
        --sets grefcoco_val_merged,refcoco_testA,refcoco_testB --out DIR
Writes DIR/<set>.npz (one row per question, in the same order as ablation/eval_ab.py / eval.py rows) + DIR/<set>.info.json.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from arch.quality.edgeq import quality, iou_b

TEMPS = (1.0, 0.5, 2.0)


def to_dev(b, d):
    return {k: (v.to(d, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def collect(acc, out, raw, extra):
    for k, v in {**out, **raw}.items():
        acc.setdefault(k, []).append(v.float().cpu().numpy() if v.dtype not in (torch.int32, torch.int64) else v.cpu().numpy())
    for k, v in extra.items():
        acc.setdefault(k, []).extend(v)


def finish(acc, path, info):
    arr = {}
    for k, v in acc.items():
        arr[k] = np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v)
    np.savez_compressed(path, **arr)
    json.dump(info, open(path.replace(".npz", ".info.json"), "w"), indent=1)
    print("[dump]", path, {k: arr[k].shape for k in list(arr)[:4]}, json.dumps(info), flush=True)


@torch.no_grad()
def heads_pass(bh, hq_last, vis, vis_valid, centers, q_img):
    out = bh(hq_last, vis, vis_valid, centers, q_img)
    q, raw = quality(bh, hq_last, vis, centers, q_img, out["center_logits"], out["box_pred"].clamp(0, 1), temps=TEMPS)
    return out, q, raw


@torch.no_grad()
def run_ab(a):
    from ablation.common import ABConfig
    from ablation.dvcore.build import build, load_ckpt
    from ablation.dvcore.dataset import RecordDataset, collate
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ABConfig().update({k: v for k, v in sd["cfg"].items() if k in ABConfig.__dataclass_fields__})
    net, heads, proc = build(cfg, "cuda"); load_ckpt(a.ckpt, net, heads); net.eval(); heads.eval()
    for name in a.sets.split(","):
        ds = RecordDataset(f"{a.prep_dir}/eval_{name}.jsonl", a.image_root, proc, cfg, train=False)
        dl = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=8, collate_fn=lambda it: collate(it, ds.tok))
        acc, t0 = {}, time.time()
        for batch in dl:
            meta = batch.pop("meta"); b = to_dev(batch, "cuda")
            feats = net.image_features(b["pixel_values"], b["grid_thw"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hp, hq = net(b, feats)
            M = hq.shape[0]
            hq_last = hq[torch.arange(M, device=hq.device), b["q_last"]]
            B, N = b["vis_valid"].shape
            vis = hp.new_zeros(B, N, hp.shape[-1]); vis[b["vis_valid"]] = hp[b["p_img_mask"]]
            out, q, raw = heads_pass(heads, hq_last, vis, b["vis_valid"], b["centers"], b["q_img"])
            ex = dict(present=[], has_box=[], iou=[], n_targets=[], image_id=[])
            box = out["box_pred"].clamp(0, 1).float().cpu()
            k = 0
            for m in meta:
                W, H = m["orig_wh"]
                for qq in m["qs"]:
                    ex["present"].append(bool(qq["present"])); ex["n_targets"].append(qq.get("n_targets", 1))
                    ex["image_id"].append(int(m["rec"].get("image_id", -1)))
                    if qq.get("box") is not None:
                        gt = torch.tensor([qq["box"][0] / W, qq["box"][1] / H, qq["box"][2] / W, qq["box"][3] / H])
                        ex["iou"].append(float(iou_b(box[k], gt))); ex["has_box"].append(True)
                    else:
                        ex["iou"].append(-1.0); ex["has_box"].append(False)
                    k += 1
            collect(acc, dict(p_present=out["presence_logit"].float().sigmoid(), p_iou50=out["iou_logit"].float().sigmoid(),
                              box=out["box_pred"].clamp(0, 1).float(), **q), raw, ex)
        finish(acc, f"{a.out}/{name}.npz", dict(ckpt=a.ckpt, kind="ab", set=name, wall_s=time.time() - t0))


@torch.no_grad()
def run_v1(a):
    from dv.config import Config
    from dv.model.build import build, load_ckpt
    from dv.data.mixed import MixedDataset, collate_mixed, TokenBudgetSampler
    import eval as E
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__}).update(dict(grad_ckpt=False, max_q_tokens=96))
    net, heads, proc = build(cfg, "cuda"); load_ckpt(a.ckpt, net, heads); net.cfg_proc = proc; net.eval(); heads.eval()
    if a.alpha != 1.0:
        from arch.common import set_lora_scale
        set_lora_scale(net, a.alpha)
    for name in a.sets.split(","):
        recs = E.merged_grefcoco(a.prep_dir, name.split("_")[-2]) if name.endswith("_merged") else \
            E.v0_eval_to_v1(f"{a.prep_dir}/eval_{name}.jsonl")
        ds = MixedDataset(None, a.data_root, proc, cfg, train=False, recs=recs)
        sampler = TokenBudgetSampler(ds, 16384, 256, 64, shuffle=False, pool=len(ds) + 1)
        dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=8, collate_fn=lambda it: collate_mixed(it, ds.tok))
        acc, t0 = {}, time.time()
        for batch in dl:
            meta = batch.pop("meta"); b = to_dev(batch, "cuda")
            feats = net.image_features(b["pixel_values"], b["grid_thw"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hp, hq = net(b, feats)
            M = hq.shape[0]
            hq_last = hq[torch.arange(M, device=hq.device), b["q_last"]]
            B, N = b["vis_valid"].shape
            vis = hp.new_zeros(B, N, hp.shape[-1]); vis[b["vis_valid"]] = hp[b["p_img_mask"]]
            out, q, raw = heads_pass(heads.box, hq_last, vis, b["vis_valid"], b["centers"], b["q_img"])
            ex = dict(present=[], has_box=[], iou=[], n_targets=[], image_key=[])
            box = out["box_pred"].clamp(0, 1).float().cpu()
            k = 0
            for mt in meta:
                W, H = mt["orig_wh"]
                for qq in mt["qs"]:
                    ex["present"].append(bool(qq["present"])); ex["n_targets"].append(qq.get("n_targets", 1))
                    ex["image_key"].append(mt["rec"]["img"])
                    if qq.get("box") is not None:
                        gt = torch.tensor([qq["box"][0] / W, qq["box"][1] / H, qq["box"][2] / W, qq["box"][3] / H])
                        ex["iou"].append(float(iou_b(box[k], gt))); ex["has_box"].append(True)
                    else:
                        ex["iou"].append(-1.0); ex["has_box"].append(False)
                    k += 1
            collect(acc, dict(p_present=out["presence_logit"].float().sigmoid(), p_iou50=out["iou_logit"].float().sigmoid(),
                              box=out["box_pred"].clamp(0, 1).float(), **q), raw, ex)
        finish(acc, f"{a.out}/{name}.npz", dict(ckpt=a.ckpt, kind="v1", set=name, alpha=a.alpha, wall_s=time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["ab", "v1"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--data_root", default="/content/data")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    (run_ab if a.kind == "ab" else run_v1)(a)
