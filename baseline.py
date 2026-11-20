"""Generative grounding baseline (autoregressive, native 0-1000 bbox output).

python baseline.py --model Qwen/Qwen3.5-4B --n 500 --out baseline_results.json
Same image resize as our model (cfg.max_vis_tokens). Prompt asks for [x1,y1,x2,y2] in 0-1000 or "none".
"""
import argparse, json, random, re, statistics, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from dv.data.dataset import resize_dims

PROMPT = ('Locate the object described as "{expr}" in the image. Output its bounding box as [x1, y1, x2, y2] '
          'with coordinates normalized to 0-1000. If no such object is in the image, answer exactly "none".')


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def parse(txt, W, H):
    nums = re.findall(r"-?\d+(?:\.\d+)?", txt)
    if len(nums) >= 4:
        x1, y1, x2, y2 = [float(v) for v in nums[:4]]
        return [x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H]
    return None  # "none" or unparseable -> absent


def sample(path, n, pred, seed=0):
    rows = []
    for l in open(path):
        r = json.loads(l)
        for q in r["questions"]:
            if pred(q):
                rows.append((r, q))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--max_vis_tokens", type=int, default=640)
    ap.add_argument("--lat_images", type=int, default=30)
    ap.add_argument("--out", default="baseline_results.json")
    a = ap.parse_args()
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    proc = AutoProcessor.from_pretrained(a.model)
    is_chat = proc.tokenizer.chat_template is not None and "Base" not in a.model

    def load_img(r):
        img = Image.open(f"{a.image_root}/{r['file']}").convert("RGB")
        W, H = img.size
        w, h = resize_dims(W, H, a.max_vis_tokens, 2.0)
        return img.resize((w, h), Image.BICUBIC), W, H

    @torch.no_grad()
    def ask(img, exprs):
        """One generate call per expression (AR models answer one question per decode)."""
        outs = []
        for e in exprs:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(expr=e)}]}]
            if is_chat:
                text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
            else:
                text = f"<|vision_start|><|image_pad|><|vision_end|>{PROMPT.format(expr=e)}\nAnswer:"
            inp = proc(text=[text], images=[img], return_tensors="pt", images_kwargs=dict(do_resize=False)).to("cuda")
            g = model.generate(**inp, max_new_tokens=32, do_sample=False)
            outs.append(proc.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    res = dict(model=a.model, chat_template=is_chat, prompt=PROMPT, sets={})
    specs = [("refcoco_val", lambda q: q["present"]),
             ("grefcoco_val_notarget", lambda q: not q["present"])]
    for name, pred in specs:
        f = "refcoco_val" if name.startswith("refcoco") else "grefcoco_val"
        rows = sample(f"{a.prep_dir}/eval_{f}.jsonl", a.n, pred)
        hits, nones, lat, examples = [], [], [], []
        for r, q in rows:
            img, W, H = load_img(r)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            txt = ask(img, [q["text"]])[0]
            torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1000)
            box = parse(txt, W, H)
            nones.append(box is None)
            if q["present"]:
                hits.append(box is not None and iou(box, q["box"]) >= 0.5)
            if len(examples) < 10:
                examples.append(dict(expr=q["text"], out=txt, gt=q.get("box")))
        s = dict(n=len(rows), none_rate=float(np.mean(nones)), latency_ms_median_1q=statistics.median(lat),
                 examples=examples)
        if hits:
            s["acc50"] = float(np.mean(hits))
        else:
            s["N_acc"] = float(np.mean(nones))
        res["sets"][name] = s
        print(name, {k: v for k, v in s.items() if k != "examples"}, flush=True)
    # 8 questions on one image = 8 sequential decodes
    rows = sample(f"{a.prep_dir}/eval_refcoco_val.jsonl", 10 ** 9, lambda q: True)
    by_img = {}
    for r, q in rows:
        by_img.setdefault(r["image_id"], (r, []))[1].append(q["text"])
    lat8 = []
    for k, (r, ex) in list(by_img.items())[: a.lat_images]:
        img, W, H = load_img(r)
        ex = (ex * 8)[:8]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ask(img, ex)
        torch.cuda.synchronize(); lat8.append((time.perf_counter() - t0) * 1000)
    res["latency_ms_median_8q_same_image"] = statistics.median(lat8)
    res["gpu"] = torch.cuda.get_device_name()
    print("lat8", res["latency_ms_median_8q_same_image"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
