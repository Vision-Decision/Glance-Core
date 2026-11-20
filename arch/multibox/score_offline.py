"""Exp 4 (zero training): presence-gated set scoring on the multi-box pilot rows (v12b-multibox, jointnb2).

The pilot decodes a set from three separate readouts: the presence gate g (decides "any?"), a per-candidate match head
m_i (threshold), and a count head (argmax of P(n | present)); when the gate says present but no candidate passes, it
falls back to the single box. These can contradict each other. Here the set is a single thresholded factorised score:
    S = NMS{ i : g * s_i > tau },   count := |S|,   present := |S| > 0
with s_i either the candidate's heat-peak probability sigmoid(heat_i) ('gpeak') or its match probability m_i ('gmatch').
tau is chosen on the calibration split (calibmb) only. No model run: the rows already hold every candidate's
[x1,y1,x2,y2, heat logit, m_i, kappa_i], g and the count distribution.

  python arch/multibox/score_offline.py --dir <eval dir with grec_val/calibmb/cocoall_val .jsonl.gz> --out outputs/arch/multibox
"""
import argparse, collections, gzip, json, math, os
import numpy as np


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms_list(boxes, scores, thr=0.7):
    keep = []
    for i in sorted(range(len(boxes)), key=lambda i: -scores[i]):
        if all(_iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return keep


def set_tp(P, G, thr=0.5):
    pairs = sorted(((_iou(p, g), i, j) for i, p in enumerate(P) for j, g in enumerate(G)), reverse=True)
    up, ug, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in up and j not in ug:
            up.add(i); ug.add(j); tp += 1
    return tp


def sig(x):
    return 1 / (1 + math.exp(-max(-40, min(40, x))))


# --------------------------------------------------------------------------------------------- decoders (as v12b-multibox eval.py)
def dec_pilot_thr(r, thr):
    g = r["p_present"]
    if g < 0.5:
        return []
    c = [x for x in r["cands"] if x[5] >= thr]
    if not c:
        return [r["box"]]
    k = nms_list([x[:4] for x in c], [x[5] for x in c])
    return [c[i][:4] for i in k]


def dec_pilot_count(r, thr):
    if r["p_present"] < 0.5:
        return []
    c = sorted(r["cands"], key=lambda x: -x[5])
    c = [c[i] for i in nms_list([x[:4] for x in c], [x[5] for x in c])]
    n = int(np.argmax(r["count_p"])) + 1
    if n >= len(r["count_p"]):
        sel = [x for x in c if x[5] >= thr]
        c = sel if len(sel) >= n else c[:n]
    else:
        c = c[:n]
    return [x[:4] for x in c] or [r["box"]]


def score(r, kind):
    g = r["p_present"]
    if kind == "gpeak":
        return [g * sig(x[4]) for x in r["cands"]]
    if kind == "gmatch":
        return [g * x[5] for x in r["cands"]]
    if kind == "gmatchk":
        return [g * x[5] * x[6] for x in r["cands"]]
    raise ValueError(kind)


def dec_factor(r, kind, tau):
    s = score(r, kind)
    idx = [i for i, v in enumerate(s) if v > tau]
    if not idx:
        return []
    k = nms_list([r["cands"][i][:4] for i in idx], [s[i] for i in idx])
    return [r["cands"][idx[j]][:4] for j in k]


# --------------------------------------------------------------------------------------------- metrics
def gt(r):
    return (r.get("gt_boxes") or ([r["gt_box"]] if r.get("gt_box") else [])) if r["present"] else []


def per_row(rows, dec):
    """-> arrays: group (0 none / 1 single / 2 multi), f1, exact, count_ok, pred_n."""
    out = collections.defaultdict(list)
    for r in rows:
        G = gt(r)
        if r["present"] and not G:
            continue
        P = dec(r)
        tp = set_tp(P, G)
        f1 = 1.0 if (not P and not G) else 2 * tp / (len(P) + len(G))
        out["grp"].append(0 if not G else (1 if len(G) == 1 else 2))
        out["f1"].append(f1); out["exact"].append(float(f1 == 1.0)); out["count_ok"].append(float(len(P) == len(G)))
        out["pred_n"].append(len(P)); out["gate"].append(r["p_present"] >= 0.5)
        cp = r["count_p"]; g = r["p_present"]
        uncond = [1 - g] + [g * x for x in cp]                      # P(n = 0, 1, 2, 3, >=4)
        out["count_head_n"].append(int(np.argmax(uncond)))
    return {k: np.array(v) for k, v in out.items()}


def summ(A):
    s = {}
    for name, gi in (("no_target", 0), ("single", 1), ("multi", 2)):
        m = A["grp"] == gi
        s[name] = dict(n=int(m.sum()), f1=float(A["f1"][m].mean()), pr_f1_1=float(A["exact"][m].mean()), count_acc=float(A["count_ok"][m].mean()))
    s["all"] = dict(n=len(A["f1"]), f1=float(A["f1"].mean()), pr_f1_1=float(A["exact"].mean()), count_acc=float(A["count_ok"].mean()))
    s["N_acc"] = s["no_target"]["pr_f1_1"]
    pn = np.minimum(A["pred_n"], 4)
    gate_vs_set = A["gate"] != (A["pred_n"] > 0)
    cnt_vs_set = A["count_head_n"] != pn
    s["contradiction"] = dict(gate_vs_set=float(gate_vs_set.mean()), counthead_vs_set=float(cnt_vs_set.mean()),
                              any=float((gate_vs_set | cnt_vs_set).mean()))
    return s


def boot_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def load(p):
    return [json.loads(l) for l in gzip.open(p, "rt")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cal = [r for r in load(f"{a.dir}/calibmb.jsonl.gz") if r.get("kind", "box") == "box"]
    grid_m = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    grid_t = [round(x, 3) for x in np.linspace(0.05, 0.9, 35)]
    choose = {}

    def cal_obj(dec):  # objective on the calibration split: Pr@F1=1 over all rows (GREC's headline)
        return summ(per_row(cal, dec))["all"]["pr_f1_1"]

    cands = {"pilot_thr": (grid_m, lambda t: (lambda r: dec_pilot_thr(r, t))),
             "pilot_count": (grid_m, lambda t: (lambda r: dec_pilot_count(r, t))),
             "gpeak": (grid_t, lambda t: (lambda r: dec_factor(r, "gpeak", t))),
             "gmatch": (grid_t, lambda t: (lambda r: dec_factor(r, "gmatch", t))),
             "gmatchk": (grid_t, lambda t: (lambda r: dec_factor(r, "gmatchk", t)))}
    for name, (grid, mk) in cands.items():
        sc = {t: cal_obj(mk(t)) for t in grid}
        best = max(sc, key=sc.get)
        choose[name] = dict(tau=best, calib_pr=sc[best])
        print("[calib]", name, best, round(sc[best], 4), flush=True)
    res = dict(source=os.path.abspath(a.dir), calib="calibmb.jsonl.gz", choose=choose, sets={})
    for sname in ("grec_val", "cocoall_val"):
        rows = [r for r in load(f"{a.dir}/{sname}.jsonl.gz") if r.get("kind", "box") == "box"]
        A = {"as_decoded": per_row(rows, lambda r: r["boxes"])}
        for name, (_, mk) in cands.items():
            A[name] = per_row(rows, mk(choose[name]["tau"]))
        S = {k: summ(v) for k, v in A.items()}
        for k in A:
            if k != "as_decoded":
                S[k]["delta_vs_as_decoded"] = dict(
                    pr_f1_1=float(A[k]["exact"].mean() - A["as_decoded"]["exact"].mean()),
                    pr_f1_1_ci95=boot_diff(A[k]["exact"], A["as_decoded"]["exact"]),
                    f1=float(A[k]["f1"].mean() - A["as_decoded"]["f1"].mean()),
                    f1_ci95=boot_diff(A[k]["f1"], A["as_decoded"]["f1"]))
                for gi, gn in ((1, "single"), (2, "multi"), (0, "no_target")):
                    m = A[k]["grp"] == gi
                    if m.any():
                        S[k]["delta_vs_as_decoded"][f"{gn}_f1"] = float(A[k]["f1"][m].mean() - A["as_decoded"]["f1"][m].mean())
                        S[k]["delta_vs_as_decoded"][f"{gn}_f1_ci95"] = boot_diff(A[k]["f1"][m], A["as_decoded"]["f1"][m])
        # single-target: the single-box head's Acc@0.5 for reference
        single = [r for r in rows if r["present"] and len(gt(r)) == 1]
        S["single_head_acc50"] = float(np.mean([_iou(r["box"], gt(r)[0]) >= 0.5 for r in single]))
        res["sets"][sname] = S
        for k, v in S.items():
            if isinstance(v, dict) and "all" in v:
                print(sname, k, "PrF1", round(v["all"]["pr_f1_1"], 4), "Nacc", round(v["N_acc"], 4), "single", round(v["single"]["f1"], 4),
                      "multi", round(v["multi"]["f1"], 4), "contra", v["contradiction"], flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
