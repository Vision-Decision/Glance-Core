"""v1 sanity check with real weights (GPU): for text-state and image-state records, the shared-prefix forward (causal
mask, no LoRA) + TypedHead at init must reproduce HF's own next-token logits on the label tokens at the block's last
token, computed on the concatenated (prefix + block) sequence with HF's own position ids.
Also prints the base model's zero-shot answer distribution on a few blocks (sanity of the prompt layout).

python scripts/v1/test_v1_init.py [--model_id Qwen/Qwen3.5-4B] [--image /content/data/coco/train2014/...jpg]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from dv.config import Config
from dv.data.mixed import MixedDataset, collate_mixed, ChatTok, LETTERS
from dv.model.build import build
from train import run_model_v1, to_dev

ap = argparse.ArgumentParser()
ap.add_argument("--model_id", default="Qwen/Qwen3.5-4B")
ap.add_argument("--image", default="")
a = ap.parse_args()
cfg = Config().update(dict(task="v1", model_id=a.model_id, attn_mask="causal", lora_r=0, grad_ckpt=False))
net, heads, proc = build(cfg, "cuda", lora=False)
net.eval(); heads.eval()
recs = [
    dict(src="t", text="CRM note, 2025-03-04, account Hillcrest Bakeries\nCall with Dana (purchasing). They open two new "
                       "shops in May and asked us to double their weekly flour order from the first week of May onward.",
         questions=[dict(type="choice", instr="What does this note say about Hillcrest's flour orders from May?",
                         options=["Hillcrest will order less flour than now.", "Hillcrest will keep ordering about the same amount.",
                                  "Hillcrest will order more flour than now.", "The note says nothing about future order volume."],
                         target=[0, 0, 1, 0]),
                    dict(type="noul", instr="Does the customer plan to open new shops?", target=1.0),
                    dict(type="score", instr="How urgent is this request for the sales team?",
                         options=["not urgent", "somewhat urgent", "urgent", "very urgent"], target=[0, 1, 0, 0])]),
]
if a.image:
    from PIL import Image
    W, H = Image.open(a.image).size
    recs.append(dict(src="i", img=os.path.basename(a.image), wh=[W, H], max_vis=640,
                     questions=[dict(type="noul", instr="Is there a person in this image?", target=1.0),
                                dict(type="score", instr="How many people are in the image?",
                                     options=["0", "1", "2", "3", "4", "5 or more"], target=[0, 0, 1, 0, 0, 0]),
                                dict(type="box", text="the person on the left", present=True, box=[0, 0, 10, 10])]))
root = os.path.dirname(a.image) if a.image else "/"
ds = MixedDataset(None, root, proc, cfg, train=False, recs=recs)
tok = ds.tok
items = [ds[i] for i in range(len(ds))]
batch = to_dev(collate_mixed(items, tok), "cuda")
batch.pop("meta")
with torch.no_grad():
    res = run_model_v1(net, heads, batch, train=False)
ours = res["typed_logits"].float()
# reference: HF forward on each prefix+block, causal, HF's own rope index
hf = net.hf
groups = tok.label_token_groups()
k = 0
maxdiff = 0.0
for it in items:
    pre = []
    for kind, ids in it["segs"]:
        if kind == "ids":
            pre += ids
        else:
            g = it["grid_thw"]; pre += [tok.pad_img] * int(g[1] * g[2] // 4)
    for q, qt, labs in zip(it["q_ids"], it["qtype"], it["labs"]):
        if qt == 0:
            continue
        ids = torch.tensor([pre + q], device="cuda")
        kw = {}
        if "pixel_values" in it:
            kw = dict(pixel_values=it["pixel_values"].cuda().to(torch.bfloat16), image_grid_thw=it["grid_thw"][None].cuda(),
                      mm_token_type_ids=(ids == tok.pad_img).int())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = hf(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
        lg = out.logits[0, -1].float()
        ref = torch.stack([torch.logsumexp(lg[torch.tensor(groups[l], device="cuda")], 0) for l in labs])
        mine = ours[k, :len(labs)]
        d = ((ref - ref.logsumexp(0)) - (mine - mine.logsumexp(0))).abs().max().item()
        maxdiff = max(maxdiff, d)
        p = mine.softmax(0).tolist()
        print(f"q{k} type={qt} labels={len(labs)} max|dlogp|={d:.4f} p={[round(x, 3) for x in p]} "
              f"top_tok={tok.tok.decode([int(lg.argmax())])!r}")
        k += 1
print("MAXDIFF", maxdiff, "OK" if maxdiff < 0.15 else "MISMATCH")
