"""Check SharedPrefixLM == HF reference forward on (prefix + one question block), with a tiny random
Qwen3.5 config (CPU OK, uses torch GDN fallback if fla is missing). Also checks the block masks
isolate question blocks (changing block 1 must not change block 0).

python scripts/test_equivalence.py [--model_id Qwen/Qwen3.5-4B-Base] [--real]   (--real: use real weights on GPU)
"""
import argparse, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

from dv.config import Config
from dv.data.dataset import RecordDataset, collate
from dv.model.backbone import SharedPrefixLM

ap = argparse.ArgumentParser()
ap.add_argument("--model_id", default="Qwen/Qwen3.5-4B-Base")
ap.add_argument("--real", action="store_true")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
proc = AutoProcessor.from_pretrained(a.model_id)
if a.real:
    hf = AutoModelForImageTextToText.from_pretrained(a.model_id, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
else:
    c = AutoConfig.from_pretrained(a.model_id)
    t = c.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 128, 256, 8
    t.layer_types = t.layer_types[:8]
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 32
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 16, 16
    v = c.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 64, 128, 2, 128
    hf = AutoModelForImageTextToText.from_config(c, dtype=torch.float32).to(dev)
    # A_log/dt_bias random init is fine
hf.eval()

tmp = tempfile.mkdtemp()
recs = []
for i, (W, H) in enumerate([(320, 240), (200, 300)]):
    Image.fromarray((np.random.rand(H, W, 3) * 255).astype("uint8")).save(f"{tmp}/im{i}.jpg")
    recs.append(dict(file=f"im{i}.jpg", W=W, H=H, questions=[
        dict(text="the red car on the left", present=True, box=[1, 2, 50, 60]),
        dict(text="a dog", present=False, box=None),
        dict(text="man wearing a very long blue jacket near the door", present=True, box=[10, 20, 100, 120])]))
with open(f"{tmp}/r.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")

errs = {}
for mode in ["causal", "block_bidir"]:
    cfg = Config().update(dict(attn_mask=mode, max_vis_tokens=64, grad_ckpt=False))
    net = SharedPrefixLM(hf, cfg).eval()
    ds = RecordDataset(f"{tmp}/r.jsonl", tmp, proc, cfg, train=False)
    batch = collate([ds[0], ds[1]], ds.tok); batch.pop("meta")
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        feats = net.image_features(batch["pixel_values"], batch["grid_thw"])
        feats = [f.to(hf.dtype) for f in feats]
        hp, hq = net(batch, feats)
        if mode == "causal":
            # reference: HF forward on prefix + block m, plain causal
            for m in range(hq.shape[0]):
                b = int(batch["q_img"][m])
                pv = batch["p_valid"][b]
                ids = torch.cat([batch["p_ids"][b][pv], batch["q_ids"][m][batch["q_valid"][m]]])[None]
                n_pix = int(batch["grid_thw"][:b].prod(-1).sum())
                n_pix_b = int(batch["grid_thw"][b].prod())
                mm = (ids == hf.config.image_token_id).int()
                out = hf.model(input_ids=ids, pixel_values=batch["pixel_values"][n_pix:n_pix + n_pix_b],
                               image_grid_thw=batch["grid_thw"][b:b + 1], mm_token_type_ids=mm)
                ref = out.last_hidden_state[0]
                Lq = int(batch["q_valid"][m].sum())
                e_q = (ref[-Lq:] - hq[m, :Lq]).abs().max().item() / ref.abs().max().item()
                e_p = (ref[:-Lq] - hp[b][pv]).abs().max().item() / ref.abs().max().item()
                errs[f"causal_q{m}"] = (e_q, e_p)
                if a.real and m in (0, 3):
                    d = (ref[:-Lq] - hp[b][pv]).abs().max(-1).values
                    top = d.topk(5)
                    print("prefix err top pos", top.indices.tolist(), [round(x, 2) for x in top.values.tolist()],
                          "ref norm there", ref[:-Lq][top.indices].abs().max(-1).values.tolist(), "len", d.shape[0],
                          "median rel", (d / ref[:-Lq].abs().max(-1).values).median().item())
        # isolation: perturb block 1 tokens, block 0 must be unchanged
        b2 = dict(batch); qi = batch["q_ids"].clone(); qi[1, 1:4] = qi[1, 1:4] + 7; b2["q_ids"] = qi
        _, hq2 = net(b2, feats)
        errs[f"{mode}_isolation_block0"] = (hq2[0] - hq[0]).abs().max().item()
        errs[f"{mode}_block1_changed"] = (hq2[1] - hq[1]).abs().max().item()
for k, v in errs.items():
    print(k, v)
tol = 3e-2 if a.real else 1e-4
ok = all(max(v) < tol for k, v in errs.items() if k.startswith("causal_q")) and \
     all(v < 1e-5 for k, v in errs.items() if "isolation" in k) and \
     all(v > 0 for k, v in errs.items() if "changed" in k)
print("EQUIVALENCE_OK" if ok else "EQUIVALENCE_FAIL")
