"""Agent-style offline evaluation (round-3 request): risk-controlled answer/click rules (RCPS, Bates et al. 2021).

python ablation/agent_rcps_v2.py [--out outputs/ablation_v2/agent]

A query is answered (box / click) iff score > lambda, otherwise the agent abstains. Loss of a query = 1 if it is
answered and wrong (absent target, or present with a wrong answer: IoU<0.5 for boxes, click outside the GT box for
GUI). This loss is monotone in lambda, so RCPS applies: on a calibration split pick the most permissive lambda whose
Clopper-Pearson upper bound (confidence 1-delta, delta=0.1) on the marginal wrong-answer rate stays <= alpha for it
and every stricter lambda. Evaluated on the disjoint test split; 200 random 50/50 splits (image-level for RefCOCO).
Reported per (system, alpha): mean test coverage (answered share), success (answered and right, share of all), test
risk (mean, and fraction of splits where it exceeds alpha), and latency per query.
Systems without a score (generative base model: it answers unless it says "none") have one operating point: it is
certified iff its own UCB <= alpha, else the RCPS rule abstains on everything (coverage 0).

Data
  RefCOCO/gRefCOCO: outputs/ablation/<run>/rows*.json (gRefCOCO-val-merged: 10,834 present + 8,905 no-target).
  GUI: outputs/eval_v0_xdomain/rows/{ours,base}_{ss2,sspro,venus}_*.jsonl (Glance v0 vs Qwen3.5-4B generative base).
       VenusBench-GD rows include refusal_grounding items (target absent).
"""
import argparse, json, os
import numpy as np
from scipy.stats import beta as Beta

from twoknob_v2 import load as load_ref

ALPHAS = (0.02, 0.05, 0.10)
DELTA = 0.1


def ucb(k, n, delta=DELTA):
    return 1.0 if n == 0 else float(Beta.ppf(1 - delta, k + 1, n - k)) if k < n else 1.0


def rcps_lambda(score, wrong, alpha):
    o = np.argsort(-score, kind="stable")
    w = np.cumsum(wrong[o])
    n = len(score)
    u = np.where(w < n, Beta.ppf(1 - DELTA, w + 1, np.maximum(n - w, 1)), 1.0)   # UCB of risk when answering top-k
    bad = np.where(u > alpha)[0]
    best = int(bad[0]) if len(bad) else n        # answer the top-`best` (all stricter lambdas also pass)
    if best == 0:
        return np.inf
    return score[o][best - 1] - 1e-12 if best < n else -np.inf


def run_system(score, wrong, groups, alpha, reps=200, seed=0, answered=None):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    cov, succ, risk, viol = [], [], [], []
    for _ in range(reps):
        cal_g = rng.permutation(ug)[: len(ug) // 2]
        cal = np.isin(groups, cal_g); te = ~cal
        if score is None:            # fixed operating point (answers everything it does not refuse)
            ans = np.ones(len(wrong), bool) if answered is None else answered
            ok = ucb(int((wrong & ans)[cal].sum()), int(cal.sum())) <= alpha
            acc = ans[te] & ok
        else:
            lam = rcps_lambda(score[cal], wrong[cal], alpha)
            acc = score[te] > lam
        wr = wrong[te]
        r = float((acc & wr).mean())
        cov.append(acc.mean()); succ.append((acc & ~wr).mean()); risk.append(r); viol.append(r > alpha)
    return dict(coverage=float(np.mean(cov)), success=float(np.mean(succ)), risk=float(np.mean(risk)),
                violation_rate=float(np.mean(viol)), coverage_sd=float(np.std(cov)))


def gui(rows_dir, name, res_key, systems):
    """systems: list of (label, file, kind) kind in {'glance','base'}."""
    out = {}
    ids = None
    data = {}
    for lab, f, kind in systems:
        p = f"{rows_dir}/{f}.jsonl"
        if not os.path.exists(p):
            continue
        R = {json.loads(l)["id"]: json.loads(l) for l in open(p)}
        data[lab] = (R, kind)
        ids = set(R) if ids is None else ids & set(R)
    ids = sorted(ids)
    for lab, (R, kind) in data.items():
        rr = [R[i] for i in ids]
        present = np.array([r["present"] for r in rr]); hit = np.array([bool(r.get("hit")) for r in rr])
        wrong_if_answer = ~(present & hit)
        groups = np.arange(len(rr))
        ms = [r["ms"] for r in rr if r.get("ms") is not None]
        o = dict(n=len(rr), n_absent=int((~present).sum()), always_answer_wrong_rate=float(wrong_if_answer.mean()),
                 latency_ms_median=float(np.median(ms)) if ms else None, rules={})
        if kind == "glance":
            g = np.array([r["p_present"] for r in rr]); k = np.array([r["p_iou50"] for r in rr])
            scores = {"gate": g, "gate*iou": g * k, "two-knob w=.9": .9 * g + .1 * g * k}
        else:
            refused = np.array([bool(r.get("refused")) for r in rr])
            scores = {"answers unless 'none'": None}
            pp = [r.get("p_present") for r in rr]
            if all(p is not None for p in pp):
                scores["P(not none)"] = np.array(pp, float)
            wrong_fixed = wrong_if_answer.copy()
            # a refused query is not answered: no loss
            wrong_fixed[refused] = False
        for sn, sc in scores.items():
            for al in ALPHAS:
                if sc is None:
                    r = run_system(None, wrong_if_answer, groups, al, answered=~refused)
                else:
                    r = run_system(sc, wrong_if_answer, groups, al)
                o["rules"][f"{sn} @alpha={al}"] = r
        out[lab] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/ablation")
    ap.add_argument("--prep", default="data/prep/eval_grefcoco_val_merged.jsonl")
    ap.add_argument("--gui", default="outputs/eval_v0_xdomain/rows")
    ap.add_argument("--latency", default="outputs/ablation_v2/latency/latency_summary.json")
    ap.add_argument("--out", default="outputs/ablation_v2/agent")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"refcoco_grefcoco": {}, "gui": {}}
    lat = json.load(open(a.latency)) if os.path.exists(a.latency) else {}
    lat_of = {"ours": lat.get("Glance (ours)", {}).get("bs1_N8", {}).get("ms_median"),
              "ar": lat.get("AR-digits, incremental KV (our engine)", {}).get("bs1_N8", {}).get("ms_median")}
    for run in ("ours_neg25", "ar_digits_neg25", "ar_atomic100_neg25"):
        d = load_ref(a.src, run, a.prep)
        wrong = ~d["good"]
        if run.startswith("ours"):
            g, k = d["eta"], d["kap"]
            scores = {"gate": g, "gate*iou": g * k, "two-knob w=.9": .9 * g + .1 * g * k}
            l = lat_of["ours"]
        else:
            scores = {"1-P(none)": d["eta"], "margin (bias on none)": d["margin"],
                      "two-knob AR w=.9": .9 * d["eta"] + .1 * d["eta"] * d["kap"]}
            l = lat_of["ar"]
        o = dict(latency_ms_per_call_N8=l, rules={})
        for sn, sc in scores.items():
            for al in ALPHAS:
                o["rules"][f"{sn} @alpha={al}"] = run_system(sc, wrong, d["img"], al)
        if not run.startswith("ours"):
            for nm in ("greedy", "seqmap"):
                for al in ALPHAS:
                    r = run_system(None, wrong, d["img"], al, answered=d[nm])
                    o["rules"][f"{nm} (fixed) @alpha={al}"] = r
        res["refcoco_grefcoco"][run] = o
        print(run, {k: (round(v["coverage"], 3), round(v["risk"], 4)) for k, v in o["rules"].items()}, flush=True)
    for name, sys in (("screenspot_v2", [("Glance v0 @640", "ours_ss2_640", "glance"), ("Qwen3.5-4B generative @4000", "base_ss2_4000", "base")]),
                      ("screenspot_pro", [("Glance v0 @640", "ours_sspro_640", "glance"), ("Glance v0 @4000", "ours_sspro_4000", "glance"),
                                          ("Qwen3.5-4B generative @4000", "base_sspro_4000", "base")]),
                      ("venusbench_gd", [("Glance v0 @640", "ours_venus_640", "glance"), ("Glance v0 @4000", "ours_venus_4000", "glance"),
                                         ("Qwen3.5-4B generative @4000", "base_venus_4000", "base")])):
        res["gui"][name] = gui(a.gui, name, name, sys)
        for lab, o in res["gui"][name].items():
            print(name, lab, o["n"], o["latency_ms_median"], {k: (round(v["coverage"], 3), round(v["risk"], 4)) for k, v in o["rules"].items()}, flush=True)
    json.dump(res, open(f"{a.out}/agent_rcps.json", "w"), indent=1)


if __name__ == "__main__":
    main()
