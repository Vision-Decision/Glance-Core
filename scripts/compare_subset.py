"""Score our model on exactly the baseline's sampled questions (same sampler/seed as baseline.py).
python scripts/compare_subset.py ROWS_JSON [prep_dir] [n]"""
import json, random, sys
import numpy as np

rows = json.load(open(sys.argv[1]))
prep = sys.argv[2] if len(sys.argv) > 2 else "data/prep"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 500


def sample_idx(path, pred, n, seed=0):
    allq = [q for l in open(path) for q in json.loads(l)["questions"]]
    idx = [i for i, q in enumerate(allq) if pred(q)]
    pairs = [(i, allq[i]) for i in idx]
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


out = {}
for name, f, pred in [("refcoco_val", "refcoco_val", lambda q: q["present"]),
                      ("grefcoco_val_notarget", "grefcoco_val", lambda q: not q["present"])]:
    R = rows[f]
    pairs = sample_idx(f"{prep}/eval_{f}.jsonl", pred, n)
    for i, q in pairs:
        assert R[i]["text"] == q["text"], (i, R[i]["text"], q["text"])
    sel = [R[i] for i, _ in pairs]
    if name == "refcoco_val":
        out[name] = dict(n=len(sel), acc50=float(np.mean([r["iou"] >= 0.5 for r in sel])),
                         gated_acc50=float(np.mean([r["iou"] >= 0.5 and r["p_present"] >= 0.5 for r in sel])),
                         said_absent_rate=float(np.mean([r["p_present"] < 0.5 for r in sel])))
    else:
        out[name] = dict(n=len(sel), N_acc=float(np.mean([r["p_present"] < 0.5 for r in sel])))
print(json.dumps(out, indent=1))
