"""Generative grounding with the base model (default Qwen/Qwen3.5-4B; chat template, thinking off), same prompt and
image resize as baseline.py, on eval.py-format prep files. Besides the greedy answer it records a first-token presence
probability: P(first token starts a box '[') / (P('[') + P('none'/'None')), so the base's absence decisions can be
compared with ours, including confidence.
Optional --lora_ckpt: run the same generation through a Glance checkpoint's LoRA (heads unused), to test whether the
fine-tune changed the generative pathway.
Output: gzip JSONL rows {meta, text, present, out, box, gt_box, pred_present, p_present_gen, iou}.
Usage: python scripts/sprint_robust/base_gen.py --files /content/robust/eval_rg.jsonl --out_dir /content/rows/base
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
ap.add_argument("--files", required=True, help="comma list of prep jsonl files; output name = basename without eval_")
ap.add_argument("--image_root", default="/content/data/coco/train2014")
ap.add_argument("--max_vis_tokens", type=int, default=640)
ap.add_argument("--bs", type=int, default=24)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--lora_ckpt", default="")
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
if a.lora_ckpt:
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    sd = torch.load(a.lora_ckpt, map_location="cpu", weights_only=False)
    c = sd["cfg"]
    inject_adapter_in_model(LoraConfig(r=c["lora_r"], lora_alpha=c["lora_alpha"], target_modules=c["lora_target"], bias="none"), model)
    set_peft_model_state_dict(model, sd["lora"])
    model = model.to(torch.bfloat16).cuda().eval()
proc = AutoProcessor.from_pretrained(a.model)
proc.tokenizer.padding_side = "left"
tk = proc.tokenizer
first_ids = lambda words: sorted({tk(w, add_special_tokens=False)["input_ids"][0] for w in words})
BOX_T = first_ids(["[", " ["]); NONE_T = first_ids(["none", "None", " none", "NONE"])

for fn in a.files.split(","):
    name = os.path.basename(fn).replace(".jsonl", "").replace("eval_", "")
    jobs = [(r, q) for l in open(fn) for r in [json.loads(l)] for q in r["questions"]]
    if a.limit:
        jobs = jobs[: a.limit]
    out = gzip.open(f"{a.out_dir}/{name}.jsonl.gz", "wt")
    for i in range(0, len(jobs), a.bs):
        chunk = jobs[i:i + a.bs]
        imgs, texts = [], []
        for r, q in chunk:
            img = Image.open(f"{a.image_root}/{r['file']}").convert("RGB")
            w, h = resize_dims(*img.size, a.max_vis_tokens, 2.0)
            imgs.append(img.resize((w, h), Image.BICUBIC))
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(expr=q["text"])}]}]
            texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False))
        inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True, images_kwargs=dict(do_resize=False)).to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=32, do_sample=False, output_scores=True, return_dict_in_generate=True)
        p0 = g.scores[0].float().softmax(-1)
        pb, pn = p0[:, BOX_T].sum(-1), p0[:, NONE_T].sum(-1)
        for k, ((r, q), seq) in enumerate(zip(chunk, g.sequences[:, inp["input_ids"].shape[1]:])):
            txt = tk.decode(seq, skip_special_tokens=True)
            box = parse(txt, r["W"], r["H"])
            row = dict(meta=q.get("meta"), text=q["text"], present=q["present"], out=txt, box=box, gt_box=q["box"],
                       pred_present=box is not None, p_present_gen=float(pb[k] / (pb[k] + pn[k] + 1e-12)),
                       p_box_tok=float(pb[k]), p_none_tok=float(pn[k]))
            if box is not None and q["box"] is not None:
                row["iou"] = iou(box, q["box"])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        if i % (a.bs * 20) == 0:
            print(name, i, len(jobs), flush=True)
    out.close()
print("done")
