"""Subsample absent questions of data/prep/train.jsonl to a target absent ratio (positives untouched).
python ablation/make_neg_ratio.py 0.05 data/prep/train.jsonl data/prep/train_neg05.jsonl
Records left without any question keep one random absent question (record count and step count unchanged)."""
import json, random, sys

ratio, src, dst = float(sys.argv[1]), sys.argv[2], sys.argv[3]
recs = [json.loads(l) for l in open(src)]
npos = sum(q["present"] for r in recs for q in r["questions"])
nneg = sum(not q["present"] for r in recs for q in r["questions"])
keep = (ratio / (1 - ratio)) * npos / nneg
rng = random.Random(0)
out_pos = out_neg = 0
with open(dst, "w") as f:
    for r in recs:
        qs = [q for q in r["questions"] if q["present"] or rng.random() < keep]
        if not qs:
            qs = [rng.choice(r["questions"])]
        out_pos += sum(q["present"] for q in qs); out_neg += sum(not q["present"] for q in qs)
        f.write(json.dumps(dict(r, questions=qs)) + "\n")
print(dict(records=len(recs), pos=out_pos, neg=out_neg, absent_ratio=out_neg / (out_pos + out_neg), keep_frac=keep))
