"""Score sprint inference dumps (evals/sprint/infer.py) under several click-point decoders, scale ensembles and zoom.

python evals/sprint/analyze.py --runs PREFIX[,PREFIX...] [--ens A+B+C ...] [--zoom FIRST=ZOOM ...] --out summary.json

Click accuracy = point inside the GT box (ScreenSpot / ScreenSpot-Pro / VenusBench positive protocol). Decoders:
  box      center of the expected box decoded at the argmax patch       (what eval.py reports; default)
  patch    center of the argmax patch
  local    probability-weighted mean of patch centers in the 3x3 window around the argmax patch (needs maps)
  ioupick  among the top-k patches, the candidate with the largest p_patch * P(IoU>=.5); its box center
  vote     the top-k candidate box containing the most top-k patch mass; its box center
Ensembles fuse center distributions of several runs of the same benchmark (different resolutions) on the finest
grid (densities resampled by nearest neighbour, summed log-density = product of experts); the point is the fused
argmax cell, refined to the box center of the finest run's top-k candidate at that cell when there is one.
"""
import argparse, collections, gzip, json, math, os
import numpy as np


def load(prefix, maps=False):
    rows = [json.loads(l) for l in gzip.open(prefix + ".rows.jsonl.gz", "rt")]
    mp = None
    if maps and os.path.exists(prefix + ".maps.npz"):
        z = np.load(prefix + ".maps.npz")
        mp = [z[f"m{i}"].astype(np.float32) for i in range(len(rows))]
    return rows, mp


def inside(pt, g):
    return g is not None and g[0] <= pt[0] <= g[2] and g[1] <= pt[1] <= g[3]


def bc(b):
    return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]


def pt_local(r, m):
    gh, gw = r["grid"]; W, H = r["wh"]
    crop = r.get("crop")
    cw, ch = (crop[2] - crop[0], crop[3] - crop[1]) if crop else (W, H)
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    P = np.exp(m.reshape(gh, gw)); i0 = r["top"][0]["i"]; y, x = divmod(i0, gw)
    ys, xs = slice(max(0, y - 1), min(gh, y + 2)), slice(max(0, x - 1), min(gw, x + 2))
    w = P[ys, xs]; yy, xx = np.mgrid[ys, xs]
    cx = ((xx + .5) / gw * w).sum() / w.sum() * cw + ox
    cy = ((yy + .5) / gh * w).sum() / w.sum() * ch + oy
    return [cx, cy]


def pt_vote(r):
    best, bs = None, -1
    for c in r["top"]:
        s = sum(o["p"] for o in r["top"] if inside(o["c"], c["b"]))
        if s > bs:
            bs, best = s, c
    return bc(best["b"])


def decoders(r, m=None):
    d = dict(box=bc(r["box"]), patch=r["top"][0]["c"], vote=pt_vote(r),
             ioupick=bc(max(r["top"], key=lambda c: c["p"] * c["iou"])["b"]))
    if m is not None:
        d["local"] = pt_local(r, m)
    return d


def size_bucket(r):
    g = r["gt"]; W, H = r["wh"]
    a = (g[2] - g[0]) * (g[3] - g[1]) / (W * H)
    return "xs<0.01%" if a < 1e-4 else ("s<0.05%" if a < 5e-4 else ("m<0.2%" if a < 2e-3 else "l>=0.2%"))


def group_keys(r):
    m = r.get("meta") or {}
    ks = ["all", "type/" + str(m.get("ui_type") or m.get("task") or "?"), "plat/" + str(m.get("platform", "?")),
          "size/" + size_bucket(r)]
    if m.get("group"):
        ks.append("group/" + m["group"])
    return ks


def score(rows, pts):
    """pts: list of point per row (None for skipped). returns {group: acc}"""
    acc = collections.defaultdict(list)
    for r, p in zip(rows, pts):
        if not r["present"] or r["gt"] is None:
            continue
        for k in group_keys(r):
            acc[k].append(inside(p, r["gt"]))
    return {k: dict(n=len(v), acc=round(float(np.mean(v)), 4)) for k, v in sorted(acc.items())}


def refusal(rows):
    neg = [r for r in rows if not r["present"]]
    if not neg:
        return None
    return round(float(np.mean([r["p_logit"] < 0 for r in neg])), 4)


def fuse(runs, weights=None):
    """runs: list of (rows, maps) of the same benchmark (same row order by id). Product of experts on finest grid."""
    base_rows, base_maps = max(runs, key=lambda x: x[0][0]["grid"][0] * x[0][0]["grid"][1])
    idx = [{r["id"]: i for i, r in enumerate(rows)} for rows, _ in runs]
    weights = weights or [1.0] * len(runs)
    pts = []
    for bi, r in enumerate(base_rows):
        gh, gw = r["grid"]
        acc = np.zeros((gh, gw), np.float64)
        for (rows, maps), ix, w in zip(runs, idx, weights):
            j = ix[r["id"]]; q = rows[j]; h, wd = q["grid"]
            dens = maps[j].reshape(h, wd) - math.log(1.0 / (h * wd))     # log density relative to uniform
            yi = (np.arange(gh) + .5) / gh * h; xi = (np.arange(gw) + .5) / gw * wd
            acc += w * dens[np.minimum(yi.astype(int), h - 1)][:, np.minimum(xi.astype(int), wd - 1)]
        k = int(acc.argmax()); y, x = divmod(k, gw)
        W, H = r["wh"]
        pt = [(x + .5) / gw * W, (y + .5) / gh * H]
        cand = [c for c in r["top"] if c["i"] == k]
        pts.append(bc(cand[0]["b"]) if cand else pt)
    return base_rows, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="")
    ap.add_argument("--ens", nargs="*", default=[], help="PREFIX+PREFIX+... fused by product of experts")
    ap.add_argument("--zoom", nargs="*", default=[], help="FIRST=ZOOM pairs (zoom rows carry full-image coords)")
    ap.add_argument("--mzoom", nargs="*", default=[], help="FIRST=ZOOM with --zoom_cands > 1 (rows id#j)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for p in [x for x in a.runs.split(",") if x]:
        rows, maps = load(p, maps=True)
        dec = [decoders(r, maps[i] if maps else None) for i, r in enumerate(rows)]
        out = {}
        for name in dec[0]:
            out[name] = score(rows, [d[name] for d in dec])
        res[os.path.basename(p)] = dict(decoders=out, refusal_acc=refusal(rows), n=len(rows))
        if refusal(rows) is not None:  # VenusBench-GD official overall: micro-average over positive + refusal items
            ok = [inside(bc(r["box"]), r["gt"]) if r["present"] else r["p_logit"] < 0 for r in rows]
            res[os.path.basename(p)]["venus_official_overall"] = round(float(np.mean(ok)), 4)
            print("  venus official overall", res[os.path.basename(p)]["venus_official_overall"], flush=True)
        print(os.path.basename(p), {k: v["all"]["acc"] for k, v in out.items()}, "refusal", refusal(rows), flush=True)
    for e in a.ens:
        ps = e.split("+")
        runs = [load(p, maps=True) for p in ps]
        rows, pts = fuse(runs)
        s = score(rows, pts)
        res["ens:" + "+".join(os.path.basename(p) for p in ps)] = dict(fused=s)
        print("ens", e, s["all"], flush=True)
    for z in a.zoom:
        f, zz = z.split("=")
        fr, _ = load(f); zr, _ = load(zz)
        zi = {r["id"]: r for r in zr}
        out = {}
        for name in ("box", "patch", "vote", "ioupick"):
            pts = [decoders(zi[r["id"]])[name] if r["id"] in zi else bc(r["box"]) for r in fr]
            out[name] = score(fr, pts)
        res["zoom:" + os.path.basename(zz)] = dict(zoom=out["box"], decoders=out)
        print("zoom", zz, {k: v["all"]["acc"] for k, v in out.items()}, flush=True)
    for z in a.mzoom:   # multi-candidate zoom: pick one crop per question
        f, zz = z.split("=")
        fr, _ = load(f); zr, _ = load(zz)
        cands = collections.defaultdict(list)
        for r in zr:
            cands[r["id"].split("#")[0]].append(r)
        rules = dict(first=lambda c: [x for x in c if "#" not in x["id"]][0],
                     peak=lambda c: max(c, key=lambda x: x["top"][0]["p"]),
                     piou=lambda c: max(c, key=lambda x: x["top"][0]["p"] * x["p_iou"]),
                     presence=lambda c: max(c, key=lambda x: x["p_logit"]),
                     pres_peak=lambda c: max(c, key=lambda x: x["p_logit"] + math.log(max(x["top"][0]["p"], 1e-6))))
        out = {}
        for name, fn in rules.items():
            out[name] = score(fr, [bc(fn(cands[r["id"]])["box"]) for r in fr])
        out["oracle_any"] = score(fr, [next((bc(x["box"]) for x in cands[r["id"]] if inside(bc(x["box"]), r["gt"])), [-1, -1])
                                       for r in fr])
        res["mzoom:" + os.path.basename(zz)] = out
        print("mzoom", zz, {k: v["all"]["acc"] for k, v in out.items()}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
