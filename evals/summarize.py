"""Aggregate per-question rows from run_ours.py / run_baseline.py into outputs/eval_v0_xdomain/results.json.

python3 evals/summarize.py --rows outputs/eval_v0_xdomain/rows --out outputs/eval_v0_xdomain/results.json

Thresholds on P(present) were fixed on in-domain val BEFORE looking at any GUI result
(outputs/v0/eval_rows_refcocoval_grefcocoval.json, gRefCOCO val merged with RefCOCO val):
  t_val = 0.99  argmax over a 0.005 grid of (N-acc + T-acc(single))/2  (= 0.9074; N-acc 0.918, T-acc 0.897)
  t_def = 0.50  the default used for all in-domain numbers.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

T_VAL, T_DEF = 0.99, 0.5
COV = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def ece(p, y, n_bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return None
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    if y.min() == y.max():
        return None
    from scipy.stats import rankdata
    r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mean(x):
    return float(np.mean(x)) if len(x) else None


def risk_coverage(conf, correct):
    """Selective accuracy when answering only the top-c fraction by confidence."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    o = np.argsort(-conf, kind="stable")
    out = {}
    for c in COV:
        k = max(1, int(round(c * len(o))))
        out[f"{c:.1f}"] = round(float(correct[o[:k]].mean()), 4)
    # area under the accuracy-coverage curve (dense)
    cum = np.cumsum(correct[o]) / np.arange(1, len(o) + 1)
    out["aurc_acc"] = round(float(cum.mean()), 4)
    return out


def load(path):
    if not os.path.exists(path):
        return None
    rows = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:  # partially-written last line of a pulled file
            continue
        rows[r["id"]] = r  # resumed runs: keep one row per question id
    return list(rows.values())


def by(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return dict(sorted(g.items()))


# ------------------------------------------------------------------ ours
def ours_block(rows, bench):
    pos = [r for r in rows if r["present"]]
    neg = [r for r in rows if not r["present"]]
    s = dict(n=len(rows), n_pos=len(pos), n_neg=len(neg),
             vis_tokens_median=float(np.median([r["n_vis"] for r in rows])))
    hit = np.array([r["hit"] for r in pos], float)
    pp = np.array([r["p_present"] for r in pos])
    s["acc_point_in_box"] = mean(hit)                         # box-center click, ungated (official protocol)
    s["acc_patch_center"] = mean([r["hit_patch"] for r in pos])
    s["acc50_iou"] = mean([r["iou"] >= 0.5 for r in pos])
    s["mean_iou"] = mean([r["iou"] for r in pos])
    for name, t in (("t_val", T_VAL), ("t_def", T_DEF)):
        s[f"gated_acc_{name}"] = mean(hit * (pp >= t))        # must also say "present"
        s[f"false_refusal_rate_{name}"] = mean(pp < t)
    s["p_present_pos_median"] = float(np.median(pp))
    bc = np.array([r["p_present"] * r["p_iou50"] for r in pos])
    s["ece_boxconf_vs_iou50_pos"] = ece(bc, [r["iou"] >= 0.5 for r in pos])
    s["ece_boxconf_vs_hit_pos"] = ece(bc, hit)
    s["auroc_boxconf_hit_vs_miss"] = auroc(bc, hit)
    s["auroc_centerp_hit_vs_miss"] = auroc([r["center_p"] for r in pos], hit)
    s["risk_coverage_pos_boxconf"] = risk_coverage(bc, hit)
    if neg:
        y = [int(r["present"]) for r in rows]
        p = [r["p_present"] for r in rows]
        s["auroc_presence"] = auroc(p, y)
        s["ece_presence"] = ece(p, y)
        allbc = [r["p_present"] * r["p_iou50"] for r in rows]
        s["ece_boxconf_all"] = ece(allbc, [int(r["present"] and r["iou"] >= 0.5) for r in rows])
        s["ece_boxconf_vs_hit_all"] = ece(allbc, [int(r["present"] and r["hit"]) for r in rows])
        for name, t in (("t_val", T_VAL), ("t_def", T_DEF)):
            s[f"refusal_acc_{name}"] = mean([r["p_present"] < t for r in neg])
        # selective view over everything: answering a refusal item is always wrong
        s["risk_coverage_all_boxconf"] = risk_coverage(allbc, [int(r["present"] and r["hit"]) for r in rows])
        s["risk_coverage_all_ppresent"] = risk_coverage(p, [int(r["present"] and r["hit"]) for r in rows])
    return s


def correct_ours(r, t, gated):
    if not r["present"]:
        return r["p_present"] < t
    return bool(r["hit"]) and (r["p_present"] >= t or not gated)


def breakdown_ours(rows, key, t=T_VAL):
    return {k: dict(n=len(v), acc=mean([correct_ours(r, t, False) for r in v]),
                    gated_acc=mean([correct_ours(r, t, True) for r in v]),
                    p_present_median=float(np.median([r["p_present"] for r in v])))
            for k, v in by(rows, key).items()}


# ------------------------------------------------------------------ baseline
def base_block(rows):
    pos = [r for r in rows if r["present"]]
    neg = [r for r in rows if not r["present"]]
    s = dict(n=len(rows), n_pos=len(pos), n_neg=len(neg),
             acc_point_in_box=mean([bool(r["hit"]) for r in pos]),
             no_point_rate_pos=mean([r["refused"] for r in pos]),
             latency_ms_median=float(np.median([r["ms"] for r in rows])),
             latency_ms_p90=float(np.percentile([r["ms"] for r in rows], 90)),
             new_tokens_median=float(np.median([r["n_new_tokens"] for r in rows])),
             in_tokens_median=float(np.median([r["n_in_tokens"] for r in rows])))
    if neg:
        s["refusal_acc"] = mean([r["refused"] for r in neg])
        if all(r.get("p_present") is not None for r in rows):
            y = [int(r["present"]) for r in rows]
            p = [r["p_present"] for r in rows]
            s["auroc_presence_tokenprob"] = auroc(p, y)
            s["ece_presence_tokenprob"] = ece(p, y)
            s["risk_coverage_all_ppresent"] = risk_coverage(p, [int(r["present"] and bool(r["hit"])) for r in rows])
    return s


def correct_base(r):
    return r["refused"] if not r["present"] else bool(r["hit"])


def breakdown_base(rows, key):
    return {k: dict(n=len(v), acc=mean([correct_base(r) for r in v])) for k, v in by(rows, key).items()}


KEYS = {
    "ss2": [("platform_x_type", lambda r: f'{r["meta"]["platform"]}/{r["meta"]["ui_type"]}'),
            ("platform", lambda r: r["meta"]["platform"])],
    "sspro": [("group_x_type", lambda r: f'{r["meta"]["group"]}/{r["meta"]["ui_type"]}'),
              ("group", lambda r: r["meta"]["group"]), ("ui_type", lambda r: r["meta"]["ui_type"])],
    "venus": [("task", lambda r: r["meta"]["task"]), ("platform", lambda r: r["meta"]["platform"])],
}


def failures(rows, bench, k=4):
    ex = []
    pos = [r for r in rows if r["present"]]
    miss = sorted([r for r in pos if not r["hit"]], key=lambda r: -r["p_present"] * r["p_iou50"])
    for r in miss[:k]:
        ex.append(dict(kind="confident_miss", id=r["id"], text=r["text"], gt=r["gt"], pred_box=r["box"],
                       point=r["point"], wh=r["wh"], p_present=round(r["p_present"], 4), p_iou50=round(r["p_iou50"], 3)))
    neg = [r for r in rows if not r["present"]]
    for r in sorted(neg, key=lambda r: -r["p_present"])[:k]:
        ex.append(dict(kind="refusal_false_accept", id=r["id"], text=r["text"], pred_box=r["box"], wh=r["wh"],
                       p_present=round(r["p_present"], 4)))
    for r in sorted(pos, key=lambda r: r["p_present"])[:k]:
        ex.append(dict(kind="false_refusal", id=r["id"], text=r["text"], gt=r["gt"], wh=r["wh"],
                       p_present=round(r["p_present"], 4), hit=r["hit"]))
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="outputs/eval_v0_xdomain/rows")
    ap.add_argument("--out", default="outputs/eval_v0_xdomain/results.json")
    a = ap.parse_args()
    res = dict(thresholds=dict(t_val=T_VAL, t_def=T_DEF, rule=__doc__.split("Thresholds")[1].strip()),
               ours={}, baseline={}, paired={}, latency={}, failures={})
    for bench in ["ss2", "sspro", "venus"]:
        for f in sorted(glob.glob(f"{a.rows}/ours_{bench}_*.jsonl")):
            T = f.rsplit("_", 1)[1].split(".")[0]
            rows = load(f)
            s = ours_block(rows, bench)
            for name, key in KEYS[bench]:
                s[f"by_{name}"] = breakdown_ours(rows, key)
            if bench == "venus":
                en = rows
                s["micro_avg_ungated_t_val"] = mean([correct_ours(r, T_VAL, False) for r in en])
                s["micro_avg_gated_t_val"] = mean([correct_ours(r, T_VAL, True) for r in en])
                s["micro_avg_gated_t_def"] = mean([correct_ours(r, T_DEF, True) for r in en])
                s["micro_avg_ungated_t_def"] = mean([correct_ours(r, T_DEF, False) for r in en])
            res["ours"][f"{bench}@{T}"] = s
            res["failures"][f"{bench}@{T}"] = failures(rows, bench)
            lat = f + ".latency.json"
            if os.path.exists(lat):
                res["latency"][f"ours_{bench}@{T}"] = json.load(open(lat))
        for fb in sorted(glob.glob(f"{a.rows}/base_{bench}_*.jsonl")):
            BT = fb.rsplit("_", 1)[1].split(".")[0]
            brows = load(fb)
            if brows:
                s = base_block(brows)
                for name, key in KEYS[bench]:
                    s[f"by_{name}"] = breakdown_base(brows, key)
                if bench == "venus":
                    s["micro_avg"] = mean([correct_base(r) for r in brows])
                res["baseline"][f"{bench}@{BT}"] = s
                res["latency"][f"baseline_{bench}@{BT}"] = dict(ms_median=s["latency_ms_median"], ms_p90=s["latency_ms_p90"])
                # paired comparison on the exact items the baseline ran
                ids = {r["id"] for r in brows}
                for T in ("640", "4000"):
                    orows = load(f"{a.rows}/ours_{bench}_{T}.jsonl")
                    if not orows:
                        continue
                    om = {r["id"]: r for r in orows if r["id"] in ids}
                    common = [r for r in brows if r["id"] in om]
                    p = dict(n=len(common), baseline_acc=mean([correct_base(r) for r in common]),
                             ours_acc_ungated=mean([correct_ours(om[r["id"]], T_VAL, False) for r in common]),
                             ours_acc_gated_t_val=mean([correct_ours(om[r["id"]], T_VAL, True) for r in common]),
                             ours_acc_gated_t_def=mean([correct_ours(om[r["id"]], T_DEF, True) for r in common]))
                    b = np.array([correct_base(r) for r in common]); o = np.array([correct_ours(om[r["id"]], T_VAL, True) for r in common])
                    p["both"] = int((b & o).sum()); p["only_baseline"] = int((b & ~o).sum()); p["only_ours"] = int((~b & o).sum())
                    res["paired"][f"{bench}: ours@{T} vs baseline@{BT}"] = p
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
