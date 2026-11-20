"""Leakage check: COCO image ids of the augmented training pool / caption LM data vs every eval prep file (RefCOCO/+/g
val+test, gRefCOCO val/test, robust-groups, glance-robust-ref), and exact-text overlap of training rewrites with eval
queries.   Usage: python check_leak.py --train <aug pool jsonl> [--lm lm_captions.jsonl]"""
import argparse, glob, json, re

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--lm", default="outputs/sprint_robust/data/lm_captions.jsonl")
a = ap.parse_args()
iid = lambda p: int(re.search(r"(\d{12})", p).group(1))
ev_imgs, ev_txt = set(), {}
for f in glob.glob("data/prep/eval_*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        ev_imgs.add(r["image_id"])
        for q in r["questions"]:
            ev_txt.setdefault(q["text"].strip().lower(), f.split("/")[-1])
tr_imgs, tr_txt = set(), set()
for l in open(a.train):
    r = json.loads(l)
    tr_imgs.add(r["image_id"])
    for q in r["questions"]:
        if q.get("aug"):
            tr_txt.add(q["text"].strip().lower())
lm_imgs = {iid(json.loads(l)["img"]) for l in open(a.lm)}
rob_txt = {t for t, f in ev_txt.items() if "robust" in f or "_rg" in f}
print(dict(eval_images=len(ev_imgs), train_images=len(tr_imgs), train_eval_image_overlap=len(tr_imgs & ev_imgs),
           lm_images=len(lm_imgs), lm_eval_image_overlap=len(lm_imgs & ev_imgs),
           shifted_train_texts=len(tr_txt), shifted_eval_texts=len(rob_txt), exact_text_overlap=len(tr_txt & rob_txt)))
print("examples of identical strings (generic short phrases expected):", sorted(tr_txt & rob_txt)[:40])
