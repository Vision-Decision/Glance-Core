"""Join source items + generated text, filter, write {out}/data/{zh,para,func,vague}.jsonl (+ stats.json).
Filters: LLM ok=false; empty text; paraphrase identical to the original (case/punctuation-insensitive); func queries
that name the target / absent category (EN word match or ZH substring against a DeepSeek-built name+synonym lexicon);
vague queries that use the exact category name; absent func/vague capped at --n_abs.
Usage: python3 finalize.py --src src.jsonl --gen gen.jsonl --names cat_names.json --out <dir>
"""
import argparse, collections, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--gen", required=True)
ap.add_argument("--names", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_abs", type=int, default=500)
a = ap.parse_args()
src = [json.loads(l) for l in open(a.src)]
gen = {}
for l in open(a.gen):
    g = json.loads(l)
    gen[g["id"]] = g
names = json.load(open(a.names))
zh = lambda w: bool(re.search(r"[一-鿿]", w))


def mentions(text, words):
    t = text.lower()
    for w in words:
        w = w.lower().strip()
        if not w:
            continue
        if zh(w):
            if w in t:
                return w
        elif re.search(r"\b" + re.escape(w) + r"(s|es)?\b", t):
            return w
    return None


norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
drop = collections.Counter(); out = collections.defaultdict(list); n_abs = collections.Counter()
for it in src:
    g = gen.get(it["id"])
    if g is None:
        drop["no_generation"] += 1; continue
    if not g["ok"] or not g["text"]:
        drop[f"llm_not_ok/{it['type']}"] += 1; continue
    t = g["text"]
    if it["type"] == "para" and norm(t) == norm(it["orig"]):
        drop["para_identical"] += 1; continue
    if it["type"] == "func":
        c = it["cat"] if it["present"] else it["absent_cat"]
        w = mentions(t, names.get(c, [c]) + [c])
        if w:
            drop[f"func_names_category/{it['sub']}"] += 1; it.setdefault("_dropped", w); continue
    if it["type"] == "vague" and it["present"]:
        w = mentions(t, [it["cat"]] + [x for x in names.get(it["cat"], []) if zh(x) and len(x) >= 2])
        if w:
            drop[f"vague_names_category/{it['sub']}"] += 1; continue
    if not it["present"] and it["type"] in ("func", "vague"):
        if n_abs[it["type"]] >= a.n_abs:
            continue
        n_abs[it["type"]] += 1
    keep = {k: v for k, v in it.items() if not k.startswith("_")}
    keep["text"] = t; keep["gen_model"] = g["model"]
    out[it["type"]].append(keep)
os.makedirs(f"{a.out}/data", exist_ok=True)
stats = dict(dropped=dict(drop), counts={})
for typ, v in out.items():
    with open(f"{a.out}/data/{typ}.jsonl", "w") as f:
        for it in v:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    stats["counts"][typ] = dict(collections.Counter(f"{it['sub']}/{it['lang']}/{'present' if it['present'] else 'absent'}" for it in v))
    stats["counts"][typ]["total"] = len(v)
json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(stats, indent=1, ensure_ascii=False))
