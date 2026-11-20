"""Base-generated GUI replay (SDFT-style, phase 2): the BASE Qwen3.5-4B answers GUI click questions from the training
mixture with the official ScreenSpot-Pro Qwen3.5 click prompt; answers whose point falls inside the GT box are kept as
LM-loss targets in the base model's own words (train.py CaptionLM: w_lm > 0, lm_file = this output). Training on them
keeps the LoRA'd backbone's generative GUI grounding close to the base (the 59.9 ScreenSpot-Pro ability).

  python -m arch.train.make_replay --train /content/data/prep_arch/train_base_avail.jsonl --n 4000 --max_vis 4096 \
      --out /content/data/prep_arch/gui_replay.jsonl
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from dv.data.dataset import resize_dims
from evals.run_baseline import CLICK_SYS, CLICK_PREFILL, parse_point, in_box

Image.MAX_IMAGE_PIXELS = None


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--data_root", default="/content/data")
    ap.add_argument("--n", type=int, default=4000, help="target number of kept samples")
    ap.add_argument("--max_vis", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.train)]
    items = [(r, q) for r in recs if r.get("src") == "gui" and r.get("img")
             for q in r["questions"] if q.get("type", "box") == "box" and q.get("present") and q.get("box")]
    random.Random(0).shuffle(items)
    m = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
    kept = tried = 0
    f = open(a.out, "w")
    for r, q in items:
        if kept >= a.n:
            break
        img = Image.open(f"{a.data_root}/{r['img']}").convert("RGB")
        W, H = img.size
        sx, sy = W / r["wh"][0], H / r["wh"][1]
        w, h = resize_dims(W, H, a.max_vis, 1.0)
        msgs = [{"role": "system", "content": [{"type": "text", "text": CLICK_SYS}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q["text"]}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False) + CLICK_PREFILL
        inp = proc(text=[text], images=[img.resize((w, h), Image.BICUBIC)], return_tensors="pt", images_kwargs=dict(do_resize=False)).to("cuda")
        g = m.generate(**inp, max_new_tokens=32, do_sample=False)
        out = proc.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        tried += 1
        pt = parse_point(out, True)
        b = [q["box"][0] * sx, q["box"][1] * sy, q["box"][2] * sx, q["box"][3] * sy]
        if pt and in_box([pt[0] * W, pt[1] * H], b):
            cap = out.split("</tool_call>")[0] + "</tool_call>" if "</tool_call>" in out else out
            f.write(json.dumps(dict(img=r["img"], system=CLICK_SYS, prompt=q["text"], prefill=CLICK_PREFILL, caption=cap,
                                    sub=r.get("sub"))) + "\n")
            kept += 1
        if tried % 200 == 0:
            print(f"[replay] tried {tried} kept {kept}", flush=True); f.flush()
    print(f"[replay] DONE tried {tried} kept {kept} acc {kept / max(1, tried):.3f}", flush=True)


if __name__ == "__main__":
    main()
