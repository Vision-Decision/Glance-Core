"""glance-robust-ref items -> eval.py prep files ({out}/eval_robust_{type}.jsonl, one line per image).
Each question carries meta={id,type,sub,lang} so eval.py --rows_dir rows can be joined back (v1 rows keep meta;
v0 rows are in file order). Usage: python make_eval.py --data <dir with {type}.jsonl> --out /content/robust
"""
import argparse, collections, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for fn in sorted(glob.glob(f"{a.data}/*.jsonl")):
    typ = os.path.basename(fn)[:-6]
    by = collections.OrderedDict()
    for l in open(fn):
        it = json.loads(l)
        r = by.setdefault(it["image_id"], dict(image_id=it["image_id"], file=it["file"], W=it["W"], H=it["H"], questions=[]))
        r["questions"].append(dict(text=it["text"], present=it["present"], box=it["box"], n_targets=int(it["present"]),
                                   boxes=[it["box"]] if it["present"] else [],
                                   meta=dict(id=it["id"], type=it["type"], sub=it["sub"], lang=it["lang"])))
    with open(f"{a.out}/eval_robust_{typ}.jsonl", "w") as f:
        for r in by.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(typ, len(by), "images", sum(len(r["questions"]) for r in by.values()), "questions")
