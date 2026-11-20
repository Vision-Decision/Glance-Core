"""Paired comparison of two checkpoints' rows on the same items (image-cluster paired bootstrap).

python scripts/sprint_abs/compare.py --a <rows dir A> --b <rows dir B> --names v1,hnft --out X.json [--sets ...]
Metrics per set (raw outputs, no recalibration, plus each model's own calib_natural/guisyn-fit Platt where noted):
presence AUROC, N-acc@0.5 (absent items), recall@0.5 (present), Acc@0.5 / click acc on present items, κ̂ AUROC,
presence ECE (raw). B - A differences with 95% CIs."""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_rows, sig, auroc, ece, cluster_boot  # noqa: E402
from analyze import best_iou, click_ok, group_id  # noqa: E402

GUI = {"ss2", "venus", "guisyn"}


def metrics(name, rows):
    y = np.array([r["present"] for r in rows], float)
    p = sig(np.array([r["p_logit"] for r in rows]))
    k = sig(np.array([r["iou_logit"] for r in rows]))
    hasb = np.array([r["present"] and r.get("gt_box") is not None for r in rows])
    ok = np.array([(click_ok(r) if name in GUI else best_iou(r) >= 0.5) if h else 0 for r, h in zip(rows, hasb)], float)
    fns = {}
    if (y == 0).any() and (y == 1).any():
        fns["auroc"] = lambda i: auroc(p[i], y[i])
        fns["N_acc"] = lambda i: float((p[i][y[i] == 0] < 0.5).mean())
        fns["ece_presence"] = lambda i: ece(p[i], y[i])
    if (y == 1).any():
        fns["recall"] = lambda i: float((p[i][y[i] == 1] >= 0.5).mean())
    if hasb.any():
        fns["acc"] = lambda i: float(ok[i][hasb[i]].mean())
        fns["kappa_auroc"] = lambda i: auroc(k[i][hasb[i]], ok[i][hasb[i]])
    return fns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--names", default="A,B")
    ap.add_argument("--sets", default=""); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    na, nb = a.names.split(",")
    sets = a.sets.split(",") if a.sets else sorted(f[:-9] for f in os.listdir(a.b) if f.endswith(".jsonl.gz") and os.path.exists(os.path.join(a.a, f)))
    res = {}
    for s in sets:
        A = load_rows(os.path.join(a.a, s + ".jsonl.gz")); B = load_rows(os.path.join(a.b, s + ".jsonl.gz"))
        key = lambda r: (r.get("img") or "", r["text"], json.dumps(r.get("gt_box")))
        ia = {key(r): r for r in A}
        pairs = [(ia[key(r)], r) for r in B if key(r) in ia]
        if s == "guisyn":
            from analyze import guisyn_split
            pairs = [(x, y) for x, y in pairs if guisyn_split(x) == "test"]
        A2, B2 = [x for x, _ in pairs], [y for _, y in pairs]
        fa, fb = metrics(s, A2), metrics(s, B2)
        groups = [group_id(r) for r in A2]
        d = dict(n=len(pairs))
        for m in fa:
            d[m] = dict(**{na: fa[m](np.arange(len(A2))), nb: fb[m](np.arange(len(B2)))},
                        diff=cluster_boot(groups, lambda i: fb[m](i) - fa[m](i), B=a.boot))
        res[s] = d
        print(s, {m: (round(v[na], 4), round(v[nb], 4), round(v["diff"]["lo"], 4), round(v["diff"]["hi"], 4)) for m, v in d.items() if m != "n"}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
