"""dv.glancex == upstream glancex on a tiny random Qwen3.5 (CPU or GPU, fp32; no checkpoint needed).

  python scripts/glancex/test_dv_tiny.py [--model_id Qwen/Qwen3.5-4B-Base] [--device cpu]   (needs `glancex` importable)

Same HF weights, same head weights, same batch: for K in {0, 4} and every direction, the readout, the image-token
states, every head output and the multi-box decode must match (max |diff| is printed; prints ALL_OK if all are 0).
The training-target helpers (obj_target, union_center_target) and the loss are compared on the same batch.
"""
import argparse, json, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

import glancex.backbone as up_bb, glancex.heads as up_heads, glancex.data as up_data, glancex.train as up_run  # noqa: E401
from glancex.config import Config as UpConfig
import dv.glancex.backbone as dv_bb, dv.glancex.heads as dv_heads, dv.glancex.data as dv_data, dv.glancex.run as dv_run  # noqa: E401
from dv.glancex.config import Config as DvConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model_id", default="Qwen/Qwen3.5-4B-Base")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
dev = a.device
torch.manual_seed(0)
proc = AutoProcessor.from_pretrained(a.model_id)
c = AutoConfig.from_pretrained(a.model_id)
t = c.text_config
t.hidden_size, t.intermediate_size, t.num_hidden_layers = 128, 256, 8
t.layer_types = t.layer_types[:8]
t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 32
t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 16, 16
v = c.vision_config
v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 64, 128, 2, 128
hf = AutoModelForImageTextToText.from_config(c, dtype=torch.float32).to(dev).eval()

tmp = tempfile.mkdtemp()
with open(f"{tmp}/r.jsonl", "w") as f:
    for i, (W, H) in enumerate([(320, 240), (200, 300)]):
        Image.fromarray((np.random.rand(H, W, 3) * 255).astype("uint8")).save(f"{tmp}/im{i}.jpg")
        f.write(json.dumps(dict(image_id=i, file=f"im{i}.jpg", W=W, H=H, questions=[
            dict(text="the red car on the left", present=True, box=[1, 2, 50, 60]),
            dict(text="a dog", present=False, box=None),
            dict(text="two people", present=True, boxes=[[10, 20, 100, 120], [120, 20, 180, 140]])])) + "\n")

res, ok = {}, True


def maxdiff(x, y):
    if isinstance(x, (list, tuple)):
        return max([maxdiff(u, w) for u, w in zip(x, y)] + [0.0])
    if torch.is_tensor(x):
        if x.shape != y.shape:
            return float("inf")
        return float((x.float() - y.float()).abs().max()) if x.numel() else 0.0
    return 0.0 if x == y else float("inf")


for K in (0, 4):
    over = dict(bidir_k=K, max_vis_tokens=64, grad_ckpt=False, pad_img_to=0, pad_txt_to=0, pad_txt_block_to=0,
                pad_t2i_block_to=0, pad_bi_block_to=0)
    ucfg, dcfg = UpConfig().update(over), DvConfig().update(over)
    unet, dnet = up_bb.GlanceXBackbone(hf, ucfg).eval(), dv_bb.GlanceXBackbone(hf, dcfg).eval()
    torch.manual_seed(1)
    uh = up_heads.GlanceXHeads(unet.hidden, ucfg).to(dev).eval()
    dh = dv_heads.GlanceXHeads(dnet.hidden, dcfg).to(dev).eval()
    missing = dh.load_state_dict(uh.state_dict(), strict=True)
    uds = up_data.RecordDataset(f"{tmp}/r.jsonl", tmp, proc, ucfg, train=False)
    dds = dv_data.RecordDataset(f"{tmp}/r.jsonl", tmp, proc, dcfg, train=False)
    for mode in ("i2t", "t2i", "bi"):
        torch.manual_seed(2)
        ub = up_data.collate([uds[0], uds[1]], uds.tok, ucfg, mode); ub.pop("meta")
        torch.manual_seed(2)
        db = dv_data.collate([dds[0], dds[1]], dds.tok, dcfg, mode); db.pop("meta")
        d = {"batch": max(maxdiff(ub[k], db[k]) for k in ub if torch.is_tensor(ub[k]))}
        ub, db = up_run.to_dev(ub, dev), dv_run.to_dev(db, dev)
        with torch.no_grad():
            uo, _, (uo_, urv, urc) = up_run.forward_batch(unet, uh, ub, train=False)
            do, _, (do_, drv, drc) = dv_run.forward_batch(dnet, dh, db, train=False)
            d["readout"] = maxdiff(uo_["r"], do_["r"]); d["vis"] = maxdiff(uo_["vis"], do_["vis"])
            for k in uo:
                d[f"head/{k}"] = maxdiff(uo[k], do[k])
            um = uh.decode_multi(uo_["r"], uo_["vis"], urv, urc, uo_["vis_index"], uo)
            dm = dh.decode_multi(do_["r"], do_["vis"], drv, drc, do_["vis_index"], do)
            d["decode_multi"] = max(maxdiff(x[0], y[0]) + maxdiff(x[1], y[1]) for x, y in zip(um, dm))
        # training path (targets + loss) with the same sampling seed
        torch.manual_seed(3); uo2, ut, _ = up_run.forward_batch(unet, uh, ub, train=True)
        torch.manual_seed(3); do2, dt, _ = dv_run.forward_batch(dnet, dh, db, train=True)
        ul, _ = up_heads.loss_fn(uo2, ut, ucfg); dl, _ = dv_heads.loss_fn(do2, dt, dcfg)
        d["targets"] = max(maxdiff(ut[k], dt[k]) for k in ut)
        d["loss"] = abs(float(ul) - float(dl))
        res[f"K{K}/{mode}"] = d
        ok &= all(v == 0.0 for v in d.values())
        print(f"K={K} {mode}", json.dumps({k: v for k, v in d.items()}), flush=True)
print("ALL_OK" if ok else "MISMATCH")
