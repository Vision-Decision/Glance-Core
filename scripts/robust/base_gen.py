"""Generative baseline (base Qwen3.5-4B, chat template, thinking off) on glance-robust-ref, same prompt as baseline.py.
One greedy decode per query, batched (left padding). Output: gzip JSONL rows {id, out, box, present, iou, gt_box}.
Usage: python scripts/robust/base_gen.py --prep /content/robust --sets zh,para,func,vague --out_dir /content/robust_rows/base
"""
import argparse, gzip, json, os, sys
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline import PROMPT, parse, iou  # noqa: E402
from dv.data.dataset import resize_dims  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--prep", default="/content/robust")
ap.add_argument("--sets", default="zh,para,func,vague")
ap.add_argument("--image_root", default="/content/data/coco/train2014")
ap.add_argument("--max_vis_tokens", type=int, default=640)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
proc = AutoProcessor.from_pretrained(a.model)
proc.tokenizer.padding_side = "left"

for name in a.sets.split(","):
    jobs = []
    for l in open(f"{a.prep}/eval_robust_{name}.jsonl"):
        r = json.loads(l)
        for q in r["questions"]:
            jobs.append((r, q))
    out = gzip.open(f"{a.out_dir}/robust_{name}.jsonl.gz", "wt")
    for i in range(0, len(jobs), a.bs):
        chunk = jobs[i:i + a.bs]
        imgs, texts = [], []
        for r, q in chunk:
            img = Image.open(f"{a.image_root}/{r['file']}").convert("RGB")
            W, H = img.size
            w, h = resize_dims(W, H, a.max_vis_tokens, 2.0)
            imgs.append(img.resize((w, h), Image.BICUBIC))
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(expr=q["text"])}]}]
            texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False))
        inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True, images_kwargs=dict(do_resize=False)).to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=32, do_sample=False)
        for (r, q), seq in zip(chunk, g[:, inp["input_ids"].shape[1]:]):
            txt = proc.tokenizer.decode(seq, skip_special_tokens=True)
            box = parse(txt, r["W"], r["H"])
            row = dict(meta=q["meta"], text=q["text"], present=q["present"], out=txt, box=box, gt_box=q["box"],
                       pred_present=box is not None)
            if box is not None and q["box"] is not None:
                row["iou"] = iou(box, q["box"])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        if i % (a.bs * 25) == 0:
            print(name, i, len(jobs), flush=True)
    out.close()
print("done")
