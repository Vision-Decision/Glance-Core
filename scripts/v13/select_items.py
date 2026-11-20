"""The RefCOCO-train expressions to translate (same selection as translate_zh.py): present, boxed, >= 3 words, from the v1
natural pool; shuffled with --seed, first --n.  python scripts/v13/select_items.py --pool pool_natural.jsonl --out items.jsonl"""
import argparse, json, random

ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=70000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
items, seen = [], set()
for l in open(a.pool):
    r = json.loads(l)
    for q in r["questions"]:
        t = q.get("text", "")
        if q.get("present") and q.get("box") and q.get("src", "").startswith("refcoco") and len(t.split()) >= 3:
            k = f"{r['image_id']}|{t}"
            if k not in seen:
                seen.add(k); items.append((k, r["image_id"], t))
random.Random(a.seed).shuffle(items)
with open(a.out, "w") as f:
    for k, iid, t in items[: a.n]:
        f.write(json.dumps(dict(key=k, image_id=iid, en=t)) + "\n")
print(min(a.n, len(items)), "of", len(items))
