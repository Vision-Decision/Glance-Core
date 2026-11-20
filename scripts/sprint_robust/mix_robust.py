"""Fine-tune mixture for a robustness recipe: replay of the v1 mixture (text / GUI / PixMo records, a fixed random
fraction, identical across recipes) + the augmented natural pool restricted to the recipe's variants.
--variants none -> control (original questions of the same images only).
--gui_func FILE  -> extra GUI records with functional rewrites of element instructions (aug=guifunc).
Usage: python mix_robust.py --v1_train train.jsonl --aug aug_natural.jsonl --variants zh,para --out mix.jsonl
"""
import argparse, collections, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--v1_train", default="/content/data/prep_v1/train.jsonl")
ap.add_argument("--aug", default="/content/up/aug_natural.jsonl")
ap.add_argument("--variants", default="none")
ap.add_argument("--replay_frac", type=float, default=0.25)
ap.add_argument("--gui_func", default="")
ap.add_argument("--teacher", default="", help="teacher_iou.json: drop augmented present questions the base model grounds "
                                                 "on a different object (IoU<--teacher_min) -- teacher-verified augmentation")
ap.add_argument("--teacher_min", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = random.Random(a.seed)
var = set() if a.variants == "none" else set(a.variants.split(","))
teach = json.load(open(a.teacher)) if a.teacher else {}
out, st = [], collections.Counter()
for l in open(a.v1_train):
    r = json.loads(l)
    if r.get("src") == "natural":
        continue
    if rng.random() < a.replay_frac:
        out.append(r); st["replay_" + ("img" if r.get("img") else "text")] += 1; st["replay_q"] += len(r["questions"])
for l in open(a.aug):
    r = json.loads(l)
    qs = []
    for q in r["questions"]:
        v = q.get("aug")
        if v is None or v in var or (v == "ulit" and var & {"ufunc", "ugeneric"}):
            if teach and v not in (None, "ulit") and q["present"]:
                ti = teach.get(f"{r['image_id']}|{q['text']}")
                if ti is not None and ti < a.teacher_min:
                    st["teacher_drop"] += 1; continue
            qs.append(q); st[f"q_{v}"] += 1
    if qs:
        out.append(dict(r, questions=qs))
if a.gui_func:
    for l in open(a.gui_func):
        r = json.loads(l); out.append(r); st["gui_func_q"] += len(r["questions"])
rng.shuffle(out)
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
st["records"] = len(out); st["questions"] = sum(len(r["questions"]) for r in out)
print(json.dumps(st, indent=1))
