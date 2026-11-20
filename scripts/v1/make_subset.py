"""Small mixed subset of the v1 pools (overfit test / pilot).  python scripts/v1/make_subset.py OUT n_text n_natural n_gui n_pixmo"""
import json, random, sys
out, nt, nn, ng, npx = sys.argv[1], *map(int, sys.argv[2:6])
P = "/content/data/prep_v1"
rng = random.Random(0)
recs = []
for name, n in [("text", nt), ("natural", nn), ("gui", ng), ("pixmo", npx)]:
    rs = [json.loads(l) for i, l in zip(range(20 * n + 1000), open(f"{P}/pool_{name}.jsonl"))]
    rng.shuffle(rs)
    recs += rs[:n]
rng.shuffle(recs)
with open(out, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(out, len(recs), "records", sum(len(r["questions"]) for r in recs), "questions")
