"""Generative Qwen3.5-4B baseline on GUI benchmarks (one generate call per question, greedy, no thinking).

python evals/run_baseline.py --bench /content/gui/sspro.jsonl --prompt click --max_vis_tokens 4000 --out rows.jsonl

Prompts
  click  : the official ScreenSpot-Pro Qwen3.5 prompt (models/qwen3_5.py in likaixin2000/ScreenSpot-Pro-GUI-Grounding):
           computer_use tool, 1000x1000 screen, "If you are not sure ... guess". Assistant turn is pre-filled with the
           tool-call prefix up to `"coordinate": [` exactly like their vLLM wrapper (GUIDED_PROMPT); enable_thinking=False.
  venus  : the official VenusBench-GD Qwen3-VL prompt (models/qwen3vl_4b.py): "[x,y]" in 0-1000, "[-1,-1]" if the task
           is infeasible. Refusal = no parsable point, or a point outside [0,1000]^2 (i.e. [-1,-1]).
           Also records P(present) = P(first coordinate token is a digit) / P(digit or '-') at the first numeric step,
           a threshold-free score for AUROC.
Same image resize rule as our model (resize_dims, multiples of 32, <= max_vis_tokens, no upscale beyond max_upscale).
"""
import argparse, json, os, random, re, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from dv.data.dataset import resize_dims

Image.MAX_IMAGE_PIXELS = None

CLICK_SYS = """You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\\n* The screen's resolution is 1000x1000.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \\n* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
CLICK_PREFILL = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['

VENUS_USER = ('Output the center point of the position corresponding to the instruction: {}. The output should just be the '
              'coordinates of a point, in the format [x,y]. Additionally, if you think the task is infeasible (e.g., the '
              'task is not related to the image), the output should be [-1,-1].')


def in_box(pt, b):
    return b is not None and pt is not None and b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def parse_point(txt, prefilled):
    if prefilled:  # text continues `"coordinate": [` -> numbers up to the closing bracket
        txt = txt.split("]")[0]
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", txt)]
    if len(nums) >= 4:
        # a box: the official Venus/SS-Pro parsers take its center
        nums = [(nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2]
    if len(nums) < 2:
        return None
    x, y = nums[0] / 1000, nums[1] / 1000
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    return [x, y]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--prompt", choices=["click", "venus"], required=True)
    ap.add_argument("--max_vis_tokens", type=int, default=4000)
    ap.add_argument("--max_upscale", type=float, default=2.0)
    ap.add_argument("--ids", default="", help="optional json list of question ids to run (subset)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle_seed", type=int, default=-1,
                    help=">=0: visit questions in a seeded random order, so a run cut short is a random subset")
    ap.add_argument("--time_limit", type=float, default=0, help="stop starting new questions after this many seconds")
    a = ap.parse_args()
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    proc = AutoProcessor.from_pretrained(a.model)
    tk = proc.tokenizer
    # vocab ids whose text starts with a digit / with '-'  (for the venus P(present) score)
    vocab = tk.convert_ids_to_tokens(list(range(len(tk))))
    dig = torch.tensor([i for i, t in enumerate(vocab) if t and t.lstrip("Ġ ")[:1].isdigit()], device="cuda")
    neg = torch.tensor([i for i, t in enumerate(vocab) if t and t.lstrip("Ġ ")[:1] == "-"], device="cuda")
    keep = set(json.load(open(a.ids))) if a.ids else None

    recs = [json.loads(l) for l in open(a.bench)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    f = open(a.out, "a")
    n = 0
    t_start = time.time()
    pairs = [(r, q) for r in recs for q in r["questions"] if keep is None or q["meta"]["id"] in keep]
    if a.shuffle_seed >= 0:
        random.Random(a.shuffle_seed).shuffle(pairs)
    cur = None
    for r, q in pairs:
        if q["meta"]["id"] in done:
            continue
        if a.time_limit and time.time() - t_start > a.time_limit:
            print(f"[baseline] time limit hit after {n} questions", flush=True)
            break
        key = os.path.join(r["image_root"], r["file"])
        if cur is None or cur[0] != key:
            img = Image.open(key).convert("RGB")
            W, H = img.size
            w, h = resize_dims(W, H, a.max_vis_tokens, a.max_upscale)
            img = img.resize((w, h), Image.BICUBIC)
            cur = (key, img, W, H, w, h)
        _, img, W, H, w, h = cur
        if a.prompt == "click":
            msgs = [{"role": "system", "content": [{"type": "text", "text": CLICK_SYS}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q["text"]}]}]
            prefill = CLICK_PREFILL
        else:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VENUS_USER.format(q["text"])}]}]
            prefill = ""
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False) + prefill
        torch.cuda.synchronize(); t0 = time.perf_counter()
        inp = proc(text=[text], images=[img], return_tensors="pt", images_kwargs=dict(do_resize=False)).to("cuda")
        g = model.generate(**inp, max_new_tokens=32, do_sample=False, output_scores=True, return_dict_in_generate=True)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
        gen = g.sequences[0, inp["input_ids"].shape[1]:]
        txt = tk.decode(gen, skip_special_tokens=True)
        pt = parse_point(txt, bool(prefill))
        p_present = None
        if a.prompt == "venus":
            for step, tid in enumerate(gen.tolist()):
                s = tk.convert_ids_to_tokens(tid).lstrip("Ġ ")
                if s[:1].isdigit() or s[:1] == "-":
                    pr = g.scores[step][0].float().softmax(-1)
                    pd, pn = float(pr[dig].sum()), float(pr[neg].sum())
                    p_present = pd / max(pd + pn, 1e-12)
                    break
            if p_present is None:
                p_present = 0.0
        pt_px = [pt[0] * W, pt[1] * H] if pt else None
        row = dict(id=q["meta"]["id"], meta=q["meta"], text=q["text"], present=bool(q["present"]), gt=q["box"],
                   wh=[W, H], in_wh=[w, h], out=txt, point=pt_px, refused=pt is None, p_present=p_present,
                   hit=in_box(pt_px, q["box"]) if q["box"] is not None else None, ms=ms,
                   n_in_tokens=int(inp["input_ids"].shape[1]), n_new_tokens=int(gen.shape[0]))
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        if n % 20 == 0:
            f.flush()
            print(f"[baseline] {n} done, {(time.time()-t_start)/n:.2f}s/q", flush=True)
    f.close()
    print(f"[baseline] finished {n} in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
