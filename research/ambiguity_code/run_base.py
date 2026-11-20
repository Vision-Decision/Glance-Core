"""Generative Qwen3.5-4B baseline (greedy, thinking off) on subsets of the (A) and (B) probes. Same resize as Glance (640 visual tokens)."""
import json, sys, os, re, time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from glance.prompt import resize_dims
S, out = sys.argv[1], sys.argv[2]
nB = int(sys.argv[3]); nA = int(sys.argv[4])
PROMPT = ('Locate the object described as "{expr}" in the image. Output its bounding box as [x1, y1, x2, y2] '
          'with coordinates normalized to 0-1000. If no such object is in the image, answer exactly "none".')
dev = 'mps'
model = AutoModelForImageTextToText.from_pretrained('Qwen/Qwen3.5-4B', dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
proc = AutoProcessor.from_pretrained('Qwen/Qwen3.5-4B')
def parse(txt, W, H):
    nums = re.findall(r'-?\d+(?:\.\d+)?', txt)
    if len(nums) >= 4:
        x1, y1, x2, y2 = [float(v) for v in nums[:4]]
        return [x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H]
    return None
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    U = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - I; return I / U if U > 0 else 0.0
@torch.no_grad()
def ask(path, e):
    img = Image.open(path).convert('RGB'); W, H = img.size
    w, h = resize_dims(W, H, 640, 2.0); img = img.resize((w, h), Image.BICUBIC)
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT.format(expr=e)}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    inp = proc(text=[text], images=[img], return_tensors='pt', images_kwargs=dict(do_resize=False)).to(dev)
    g = model.generate(**inp, max_new_tokens=32, do_sample=False)
    t = proc.tokenizer.decode(g[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
    return t, parse(t, W, H)
t0 = time.time(); resB, resA = [], []
B = json.load(open(S + '/amb/b_spec.json'))[::max(1, 78 // nB)][:nB]
for it in B:
    p = S + '/amb/img/%012d.jpg' % it['image_id']
    for q in it['questions']:
        t, b = ask(p, q['text'])
        resB.append(dict(image_id=it['image_id'], salience=it['salience'], type=q['type'], text=q['text'], present=q['present'], swap=q.get('swap'),
                         out=t, none=b is None, iou_gt=round(iou(b, it['gt_box']), 3) if b else None))
    print('B', it['idx'], round(time.time() - t0), flush=True)
A = json.load(open(S + '/amb/a_sel.json'))
Aitems = [('multi', r) for i, r in enumerate(A['multi']) if i != 34][:nA] + [('single', r) for i, r in enumerate(A['single']) if i + 40 not in (50, 51, 57)][:nA]
PL = {'person': 'people', 'bus': 'buses', 'sheep': 'sheep', 'teddy bear': 'teddy bears'}
for kind, r in Aitems:
    p = S + '/amb/imgA/%012d.jpg' % r['image_id']; Bx = r['boxes']
    U = [min(b[0] for b in Bx), min(b[1] for b in Bx), max(b[2] for b in Bx), max(b[3] for b in Bx)]
    for tag, e in (('bare_the', f"the {r['cat']}"), ('plural', f"the {PL.get(r['cat'], r['cat'] + 's')}")):
        t, b = ask(p, e)
        resA.append(dict(kind=kind, image_id=r['image_id'], n_inst=len(Bx), tag=tag, text=e, out=t, none=b is None,
                         iou_inst=[round(iou(b, g), 3) for g in Bx] if b else None, iou_union=round(iou(b, U), 3) if b else None))
    print('A', r['image_id'], round(time.time() - t0), flush=True)
json.dump(dict(model='Qwen/Qwen3.5-4B (generative, bf16, MPS, greedy, thinking off)', prompt=PROMPT, wall_s=round(time.time() - t0),
               B=resB, A=resA), open(out, 'w'), indent=0)
