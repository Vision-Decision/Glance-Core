"""Caption LM data for the w_lm loss: COCO captions of TRAIN-pool images (pool_natural.jsonl images; every RefCOCO/
gRefCOCO eval image was already removed from that pool) + DeepSeek Chinese translations of a subset.
Usage: python make_captions.py --captions captions_train2014.json --out lm_captions.jsonl [--n 6000 --n_zh 3000]
"""
import argparse, json, os, random, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import chat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--captions", required=True)
ap.add_argument("--pool", default="~/Developer/decision-vison/data/prep_v1/pool_natural.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--n_zh", type=int, default=3000)
a = ap.parse_args()
rng = random.Random(0)
imgs = {}
for l in open(a.pool):
    r = json.loads(l)
    imgs[r["image_id"]] = r["img"]
cap = json.load(open(a.captions))
by = {}
for c in cap["annotations"]:
    if c["image_id"] in imgs:
        by.setdefault(c["image_id"], []).append(" ".join(c["caption"].split()))
ids = sorted(by); rng.shuffle(ids)
ids = ids[: a.n]
EN = ["Describe this image in one sentence.", "What is in this picture?", "Write a short caption for this image."]
ZH = ["用一句话描述这张图片。", "图片里有什么？", "给这张图写一句简短的说明。"]
rows = [dict(image_id=i, img=imgs[i], caption=rng.choice(by[i])) for i in ids]
zh_rows = rows[: a.n_zh]
batches = [zh_rows[k:k + 40] for k in range(0, len(zh_rows), 40)]


def tr(b):
    try:
        res = chat("Translate each English image caption into natural Simplified Chinese. Return ONLY JSON {\"results\": "
                   "[{\"i\": <index>, \"zh\": ...}]}.", json.dumps([dict(i=k, en=r["caption"]) for k, r in enumerate(b)], ensure_ascii=False),
                   temperature=0.3)
        for x in res.get("results", []):
            b[int(x["i"])]["zh"] = str(x["zh"]).strip()
    except Exception as e:
        print("fail", str(e)[:80])


with ThreadPoolExecutor(16) as ex:
    list(ex.map(tr, batches))
with open(a.out, "w") as f:
    for k, r in enumerate(rows):
        if r.get("zh"):
            f.write(json.dumps(dict(img=r["img"], prompt=rng.choice(ZH), caption=r["zh"]), ensure_ascii=False) + "\n")
        f.write(json.dumps(dict(img=r["img"], prompt=rng.choice(EN), caption=r["caption"]), ensure_ascii=False) + "\n")
print("wrote", a.out, sum(1 for _ in open(a.out)))
