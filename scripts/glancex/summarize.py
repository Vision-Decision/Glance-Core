"""Collect score.json files (+ paired bootstraps) into outputs/glancex/summary.{json,md} tables (CPU).

  python scripts/glancex/summarize.py --rows_root ROOT --out_dir outputs/glancex [--paired PAIRED_DIR]

ROOT/<tag>/<mode>/score.json, tags: released (image_first, text_first, joint), i2t_only_s0/s1, dvmix_s0, ... .
Reported Glance-X numbers (docs/RESULTS.md @73ee4ed) are embedded for the reproduction check.
"""
import argparse, glob, json, os

REFCOCO = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocoplus_testA", "refcocoplus_testB",
           "refcocog_val", "refcocog_test"]
REPORTED = {  # docs/RESULTS.md
    "image_first": dict(refcoco=[92.9, 94.8, 90.4, 89.2, 93.0, 84.7, 90.0, 90.5], mean=90.7, reject=79.2, auroc=98.1,
                        acc_single=92.9, hit_any=81.2, multi_f1=77.3,
                        testA=dict(reject=70.5, auroc=97.3, acc=92.6, hit=77.0, f1=72.7),
                        testB=dict(reject=73.1, auroc=93.4, acc=83.4, hit=74.2, f1=63.8)),
    "text_first": dict(refcoco=[92.1, 93.9, 89.5, 87.6, 91.6, 82.0, 89.3, 89.0], mean=89.4, reject=75.9, auroc=97.3,
                       acc_single=92.1, hit_any=87.5, multi_f1=78.1,
                       testA=dict(reject=65.6, auroc=96.2, acc=91.8, hit=85.9, f1=73.4),
                       testB=dict(reject=69.7, auroc=92.7, acc=82.3, hit=81.1, f1=66.5)),
    "joint": dict(refcoco=[93.1, 94.9, 90.5, 89.6, 93.1, 84.9, 90.3, 90.5], mean=90.9, reject=78.9, auroc=98.2,
                  acc_single=93.0, hit_any=89.0, multi_f1=82.7,
                  testA=dict(reject=71.3, auroc=97.4, acc=92.7, hit=87.5, f1=78.3),
                  testB=dict(reject=73.0, auroc=93.5, acc=83.7, hit=82.3, f1=72.2)),
}
REPORTED_DIAG = {  # kind -> (metric, i2t, t2i, joint)
    "short": ("acc50", 86.7, 86.4, 87.6), "medium": ("acc50", 90.8, 90.5, 91.0), "long": ("acc50", 91.3, 92.0, 92.0),
    "casual": ("acc50", 88.5, 88.2, 89.2), "cat_only": ("acc50", 85.0, 86.5, 86.7),
    "hneg_attribute": ("reject_rate", 94.9, 93.8, 95.0), "hneg_relation": ("reject_rate", 94.2, 94.2, 94.4),
    "neg_count": ("reject_rate", 99.9, 99.9, 99.9), "neg_position": ("reject_rate", 99.0, 98.2, 99.0),
    "neg_near": ("reject_rate", 99.9, 100.0, 99.9), "group": ("multi_hit_any", 78.5, 85.1, 87.1),
    "cat_all": ("multi_hit_any", 61.4, 65.0, 66.8), "gref_multi": ("multi_hit_any", 80.9, 86.4, 86.6),
    "function": ("multi_hit_any", 58.3, 59.6, 60.6),
}
MODE_COL = {"image_first": 1, "text_first": 2, "joint": 3}


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--paired", default="")
    a = ap.parse_args()
    S = {}
    for fn in sorted(glob.glob(f"{a.rows_root}/*/*/score.json")):
        tag, mode = fn.split("/")[-3], fn.split("/")[-2]
        S[f"{tag}/{mode}"] = json.load(open(fn))
    P = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{a.paired}/*.json"))} if a.paired else {}
    out = dict(scores={k: {kk: v[kk] for kk in v if kk != "rows_dir"} for k, v in S.items()}, paired=P, reported=REPORTED)
    json.dump(out, open(f"{a.out_dir}/summary.json", "w"), indent=1)
    L = []
    # RefCOCO
    L.append("### RefCOCO family, Acc@0.5 (%), ours vs reported\n")
    keys = list(S)
    L.append("| Split | " + " | ".join(keys) + " |")
    L.append("|---|" + ":-:|" * len(keys))
    for i, s in enumerate(REFCOCO + ["mean"]):
        cells = []
        for k in keys:
            v = S[k]["refcoco"].get(s)
            mode = k.split("/")[1]
            rep = (REPORTED[mode]["refcoco"][i] if s != "mean" else REPORTED[mode]["mean"]) if k.startswith("released/") and mode in REPORTED else None
            cells.append(pct(v) + (f" ({rep})" if rep is not None else ""))
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    # gRefCOCO
    L.append("\n### gRefCOCO (%), Glance-X definitions (reported in parentheses)\n")
    L.append("| Model | split | reject_rate | AUROC | Acc@0.5 single | multi_hit_any | multi_f1 | count acc |")
    L.append("|---|---|:-:|:-:|:-:|:-:|:-:|:-:|")
    for k in keys:
        mode = k.split("/")[1]
        for sp, rk in (("grefcoco_val_merged", None), ("grefcoco_testA", "testA"), ("grefcoco_testB", "testB")):
            g = S[k]["gref_glancex"].get(sp)
            if not g:
                continue
            rep = REPORTED.get(mode) if k.startswith("released/") else None
            def r(name, sub):
                if rep is None:
                    return ""
                v = rep.get(sub) if rk is None else rep[rk].get({"reject": "reject", "auroc": "auroc", "acc_single": "acc", "hit_any": "hit", "multi_f1": "f1"}[sub])
                return f" ({v})" if v is not None else ""
            L.append(f"| {k} | {sp} | {pct(g.get('reject_rate'))}{r('', 'reject')} | {pct(g.get('auroc'))}{r('', 'auroc')} | "
                     f"{pct(g.get('acc50'))}{r('', 'acc_single')} | {pct(g.get('multi_hit_any'))}{r('', 'hit_any')} | "
                     f"{pct(g.get('multi_f1'))}{r('', 'multi_f1')} | {pct(g.get('multi_count_acc'))} |")
    # ours protocol + calibration
    L.append("\n### gRefCOCO val_merged, our eval.py definitions + calibration (%)\n")
    L.append("| Model | N_acc @0.5 | T_acc single | AUROC | ECE presence raw | ECE Platt (2-fold CV) | ECE box conf | testA ECE raw → Platt(val) | testB ECE raw → Platt(val) |")
    L.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for k in keys:
        g = S[k]["gref_ours"]
        v = g.get("grefcoco_val_merged")
        if not v:
            continue
        ta, tb = g.get("grefcoco_testA", {}), g.get("grefcoco_testB", {})
        f = lambda t: f"{pct(t.get('ece_presence'))} → {pct((t.get('platt') or {}).get('ece_presence'))}" if t else "—"
        L.append(f"| {k} | {pct(v.get('N_acc'))} | {pct(v.get('T_acc_single'))} | {pct(v.get('auroc_presence'))} | {pct(v.get('ece_presence'))} | "
                 f"{pct((v.get('platt_cv2') or {}).get('ece_presence'))} | {pct(v.get('ece_box_conf'))} | {f(ta)} | {f(tb)} |")
    # GREC
    L.append("\n### Official GREC (threshold 0.7, GIoU ≥ 0.5): Pr@F1=1 / N-acc / T-acc (%)\n")
    L.append("| Model | variant | grec_val | grec_testA | grec_testB |")
    L.append("|---|---|:-:|:-:|:-:|")
    for k in keys:
        G = S[k].get("grec", {})
        if not G:
            continue
        for var in ("single_raw@0.7", "multi_raw@0.7", "multi_prod_raw@0.7", "single_platt@0.7", "multi_platt@0.7"):
            cells = []
            for sp in ("grec_val", "grec_testA", "grec_testB"):
                m = G.get(sp, {}).get(var)
                cells.append("—" if not m else f"{pct(m['pr_f1'])} / {pct(m['N_acc'])} / {pct(m['T_acc'])}")
            L.append(f"| {k} | {var} | " + " | ".join(cells) + " |")
    # diag
    L.append("\n### Glance-Data diagnostic set by kind (%), Glance-X definitions (reported in parentheses)\n")
    L.append("| kind | metric | " + " | ".join(keys) + " |")
    L.append("|---|---|" + ":-:|" * len(keys))
    for kind, (metric, *rep) in REPORTED_DIAG.items():
        cells = []
        for k in keys:
            d = S[k].get("diag", {}).get("by_kind", {}).get(kind, {})
            mode = k.split("/")[1]
            rv = rep[MODE_COL[mode] - 1] if k.startswith("released/") and mode in MODE_COL else None
            cells.append(pct(d.get(metric)) + (f" ({rv})" if rv is not None else ""))
        L.append(f"| {kind} | {metric} | " + " | ".join(cells) + " |")
    L.append("| overall | AUROC | " + " | ".join(pct(S[k].get("diag", {}).get("overall", {}).get("auroc")) for k in keys) + " |")
    L.append("| overall | multi F1 | " + " | ".join(pct(S[k].get("diag", {}).get("overall", {}).get("multi_f1")) for k in keys) + " |")
    L.append("| overall | rows | " + " | ".join(str(S[k].get("diag", {}).get("n", "—")) for k in keys) + " |")
    if P:
        L.append("\n### Paired comparisons, RefCOCO 8-split mean Acc@0.5 (image-clustered bootstrap, 95% CI)\n")
        L.append("| comparison | A | B | A − B | 95% CI |")
        L.append("|---|:-:|:-:|:-:|:-:|")
        for name, p in P.items():
            L.append(f"| {name} | {pct(p['mean_a'])} | {pct(p['mean_b'])} | {100 * p['diff']:+.2f} | [{100 * p['ci95'][0]:+.2f}, {100 * p['ci95'][1]:+.2f}] |")
    open(f"{a.out_dir}/tables.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
