"""Paired comparison of phase-2 runs vs the control on ScreenSpot-Pro (per question) and RefCOCO testA/B (per expression).
python arch/train/compare_phase2.py DIR(with <run>/a1/...) --base p0_control --out outputs/arch/phase2/results.json"""
import argparse, gzip, json, os
import numpy as np


def hits_sspro(p):
    R = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(p, "rt")}
    return {k: float(r["gt"][0] <= (r["box"][0] + r["box"][2]) / 2 <= r["gt"][2] and r["gt"][1] <= (r["box"][1] + r["box"][3]) / 2 <= r["gt"][3]) for k, r in R.items()}


def hits_coco(p):
    return [float(json.loads(l).get("iou", 0) >= 0.5) for l in gzip.open(p, "rt") if "iou" in json.loads(l)]


def ci(d, n=2000):
    d = np.asarray(d); rng = np.random.default_rng(0)
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--base", default="p0_control"); ap.add_argument("--out", required=True)
a = ap.parse_args()
runs = sorted(os.listdir(a.dir)); res = {}
B = {"sspro": hits_sspro(f"{a.dir}/{a.base}/a1/sspro.rows.jsonl.gz")}
for s in ("refcoco_testA", "refcoco_testB"):
    B[s] = hits_coco(f"{a.dir}/{a.base}/a1/{s}.jsonl.gz")
for r in runs:
    d = {}
    h = hits_sspro(f"{a.dir}/{r}/a1/sspro.rows.jsonl.gz"); ks = sorted(B["sspro"])
    x = np.array([h[k] for k in ks]); y = np.array([B["sspro"][k] for k in ks])
    d["sspro"] = round(float(x.mean()), 4); d["sspro_delta"] = round(float((x - y).mean()), 4); d["sspro_delta_ci95"] = ci(x - y)
    for s in ("refcoco_testA", "refcoco_testB"):
        x = np.array(hits_coco(f"{a.dir}/{r}/a1/{s}.jsonl.gz")); y = np.array(B[s])
        d[s] = round(float(x.mean()), 4); d[s + "_delta"] = round(float((x - y).mean()), 4); d[s + "_delta_ci95"] = ci(x - y)
    res[r] = d; print(r, d)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(dict(base=a.base, source=a.dir, runs=res), open(a.out, "w"), indent=1)
