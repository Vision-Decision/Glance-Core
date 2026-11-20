"""Summarise the generative hybrid-vs-attention runs (arch/gdn/gen_queue.sh) + Glance's needle ablation.
python arch/gdn/summarize_gen.py GEN_DIR ABLATE_JSON OUT_JSON"""
import glob, json, os, sys
import numpy as np


def ci(x):
    x = np.asarray(x, float); rng = np.random.default_rng(0)
    b = x[rng.integers(0, len(x), (1000, len(x)))].mean(1)
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


gen, abl, out = sys.argv[1:4]
res = dict(source=dict(gen=gen, ablate=abl), needle={}, sspro={})
for f in sorted(glob.glob(f"{gen}/*.jsonl")):
    name = os.path.basename(f)[:-6]
    if "_needle_" not in name and "_sspro_" not in name:
        continue
    R = [json.loads(l) for l in open(f)]
    P = [r for r in R if r["present"]]; A = [r for r in R if not r["present"]]
    ref = lambda r: r["refused"] or (r["point"] is not None and r["point"][0] < 0)
    model, kind, x = name.rsplit("_", 2)
    hit = [bool(r["hit"]) for r in P]
    d = dict(n=len(R), click_acc=round(float(np.mean(hit)), 4), click_acc_ci95=ci(hit))
    if A:
        d["absent_refusal"] = round(float(np.mean([ref(r) for r in A])), 4)
    res[kind].setdefault(model, {})[int(x)] = d
G = json.load(open(abl))["sets"]
res["needle"]["Glance-v1.1 (head readout)"] = {int(k.split("_")[1].split(".")[0]): dict(
    click_acc=v["full"]["click_acc"], N_acc=v["full"]["N_acc"], attn_noimg_click=v["attn_noimg"]["click_acc"],
    gdn_reset_click=v["gdn_reset"]["click_acc"]) for k, v in G.items() if k.startswith("needle")}
res["refcoco_ablation"] = G.get("refcoco")
json.dump(res, open(out, "w"), indent=1)
for k in ("needle", "sspro"):
    for m, v in res[k].items():
        print(k, m, {x: (d["click_acc"]) for x, d in sorted(v.items())})
