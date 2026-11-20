"""v1.1 vs v1.2-pilot table from eval.py outputs (eval.json of scripts/multibox/vm_run.sh), plus the pre-registered check.

python scripts/multibox/compare.py base=path/base/eval.json heads=path/heads/eval.json joint=path/joint/eval.json

Pre-registered success criterion (written before any pilot result was seen):
  (1) multi-target F1 on gRefCOCO val (IoU 0.5, set matching) up clearly: >= +10 points over v1.1;
  (2) single-target Acc@0.5 within 1 point of v1.1 on RefCOCO val / testA / testB and on the 8-split RefCOCO/+/g average;
  (3) N-acc (gRefCOCO val no-target, P(present) < 0.5, raw) within 1 point of v1.1.
  Secondary (reported, not gating): presence AUROC / ECE, kappa AUROC, GREC Pr@(F1=1), count accuracy, cocoall F1.
"""
import json, sys

REC = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocoplus_testA", "refcocoplus_testB",
       "refcocog_val", "refcocog_test"]


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def metrics(e):
    S = e["sets"]
    m = {}
    for s in REC:
        m[f"Acc@0.5 {s}"] = g(S, s, "acc50")
    v = [m[f"Acc@0.5 {s}"] for s in REC]
    m["Acc@0.5 RefCOCO/+/g avg (8)"] = sum(v) / len(v) if all(x is not None for x in v) else None
    m["kappa AUROC (RefCOCO val)"] = g(S, "refcoco_val", "kappa_auroc")
    gv = S.get("grec_val", {})
    m["gRef N-acc"] = g(gv, "raw", "N_acc")
    m["gRef presence AUROC"] = g(gv, "summary", "auroc_presence")
    m["gRef presence ECE"] = g(gv, "summary", "ece_presence")
    m["gRef N-acc (T-calibrated)"] = g(gv, "calibrated", "N_acc")
    mm = gv.get("multi", {})
    m["gRef multi F1"] = g(mm, "multi", "f1")
    m["gRef multi Pr@F1=1"] = g(mm, "multi", "pr_f1_1")
    m["gRef multi count acc"] = g(mm, "multi", "count_acc")
    m["gRef multi hit-any"] = g(mm, "multi", "hit_any")
    m["gRef multi recall"] = g(mm, "multi", "recall")
    m["gRef multi precision"] = g(mm, "multi", "precision")
    m["gRef multi mean #boxes"] = g(mm, "multi", "mean_pred_n")
    m["gRef single F1"] = g(mm, "single", "f1")
    m["gRef single mean #boxes"] = g(mm, "single", "mean_pred_n")
    m["gRef GREC Pr@F1=1 (all)"] = g(mm, "all", "pr_f1_1")
    m["gRef count head acc"] = g(mm, "count_head", "acc")
    m["gRef count head AUROC multi/single"] = g(mm, "count_head", "auroc_multi_vs_single")
    m["gRef per-box conf ECE"] = g(mm, "box_conf", "ece")
    m["gRef per-box conf AUROC"] = g(mm, "box_conf", "auroc")
    ca = g(S, "cocoall_val", "multi") or {}
    m["cocoall F1 (n>=1)"] = None
    if "single" in ca and "multi" in ca:
        n1, n2 = ca["single"]["n"], ca["multi"]["n"]
        m["cocoall F1 (n>=1)"] = (ca["single"]["f1"] * n1 + ca["multi"]["f1"] * n2) / (n1 + n2)
    m["cocoall single F1"] = g(ca, "single", "f1")
    m["cocoall multi F1"] = g(ca, "multi", "f1")
    m["cocoall multi count acc"] = g(ca, "multi", "count_acc")
    m["cocoall N-acc"] = g(ca, "N_acc")
    m["cocoall count head acc"] = g(ca, "count_head", "acc")
    cb = g(S, "calibmb", "multi") or {}
    m["calib_mb multi F1"] = g(cb, "multi", "f1")
    return m


def main():
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    M = {k: metrics(json.load(open(p))) for k, p in runs.items()}
    names = list(M)
    keys = list(next(iter(M.values())).keys())
    fmt = lambda x: "  -  " if x is None else f"{100 * x:6.1f}"
    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in keys:
        print(f"| {k} | " + " | ".join(fmt(M[n][k]) for n in names) + " |")
    if "base" in M:
        b = M["base"]
        for n in names:
            if n == "base":
                continue
            r = M[n]
            d = lambda k: (r[k] - b[k]) * 100 if r[k] is not None and b[k] is not None else float("nan")
            c1 = d("gRef multi F1") >= 10
            c2 = all(d(f"Acc@0.5 {s}") >= -1 for s in REC[:3]) and d("Acc@0.5 RefCOCO/+/g avg (8)") >= -1
            c3 = d("gRef N-acc") >= -1
            print(f"\n{n}: (1) multi F1 {d('gRef multi F1'):+.1f} -> {'PASS' if c1 else 'FAIL'}; "
                  f"(2) RefCOCO val/testA/testB {d('Acc@0.5 refcoco_val'):+.1f}/{d('Acc@0.5 refcoco_testA'):+.1f}/"
                  f"{d('Acc@0.5 refcoco_testB'):+.1f}, avg8 {d('Acc@0.5 RefCOCO/+/g avg (8)'):+.1f} -> {'PASS' if c2 else 'FAIL'}; "
                  f"(3) N-acc {d('gRef N-acc'):+.1f} -> {'PASS' if c3 else 'FAIL'}; overall {'PASS' if c1 and c2 and c3 else 'FAIL'}")


if __name__ == "__main__":
    main()
