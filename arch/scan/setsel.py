"""Exp 7b (CPU): set-level statistics instead of the max over clips, on the cached Glance-Video clip scan pilot
(Charades-STA test, 2 fps, 4-frame clips stride 2; 12,475 scans = 1,394 positive + 11,081 negative query x video pairs;
per-clip yes-logit of the final Glance-Video checkpoint; HF Vision-Decision/glance-train-ckpts:results/gv0/scan/
pilot_clips_gvfinal.json.gz).

(1) scan detection ("is the query anywhere in this video?"): statistic T(s_1..s_N) for
      max | top2 mean | top3 mean | smoothed max (moving average over w=3 adjacent clips) | logsumexp (soft max, t=1) |
      noisy-OR of sigmoids | mean
    AUROC positive vs negative scans; miss rate at a 10% false-alarm rate with the conformal max-stat-style calibration
    (threshold = conformal quantile of T on held-out negative scans; split by query, 50 random halves). Any statistic
    keeps the distribution-free false-alarm guarantee; the question is which one misses less.
(2) selection inside positive scans ("which clips should the planner look at?"): hit@1 (selected clip inside the
    ground-truth span) and span recall of top-k for k=3: independent top-k by score vs temporally smoothed top-k vs
    top-k with temporal NMS (diversity, HFS/BOLT-style).
CIs: bootstrap over queries (1000).

  python arch/scan/setsel.py --scans pilot_clips_gvfinal.json.gz --out outputs/arch/scan
"""
import argparse, gzip, json, math, os
import numpy as np


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def smooth(s, w=3):
    k = np.ones(w) / w
    return np.convolve(np.pad(s, (w // 2, w // 2), mode="edge"), k, mode="valid")


STATS = {
    "max": lambda s: s.max(),
    "top2_mean": lambda s: np.sort(s)[-2:].mean(),
    "top3_mean": lambda s: np.sort(s)[-3:].mean(),
    "smooth3_max": lambda s: smooth(s, 3).max(),
    "smooth5_max": lambda s: smooth(s, 5).max(),
    "logsumexp": lambda s: float(np.log(np.exp(s - s.max()).sum()) + s.max()),
    "noisy_or": lambda s: float(-np.log1p(-np.clip(1 - np.prod(1 - 1 / (1 + np.exp(-s))), 0, 1 - 1e-12))),
    "mean": lambda s: s.mean(),
}


def conformal_q(cal, alpha):
    cal = np.sort(cal); k = math.ceil((len(cal) + 1) * (1 - alpha))
    return np.inf if k > len(cal) else cal[k - 1]


def select(s, k, mode):
    if mode == "indep":
        return list(np.argsort(-s, kind="stable")[:k])
    if mode == "smooth":
        return list(np.argsort(-smooth(s, 3), kind="stable")[:k])
    # temporal NMS: greedily take the best clip, suppress neighbours within 2 clips (~2 s), repeat
    out, used = [], np.zeros(len(s), bool)
    for i in np.argsort(-s, kind="stable"):
        if not used[i]:
            out.append(i); used[max(0, i - 2): i + 3] = True
            if len(out) == k:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    D = json.load(gzip.open(a.scans))
    S = D["scans"]
    sc = [np.array(x["scores"]["g_yes"], float) for x in S]
    pos = np.array([x["pos"] for x in S]); q = np.array([x["qid"] for x in S])
    rng = np.random.default_rng(0)
    res = dict(source=a.scans, n_scans=len(S), n_pos=int(pos.sum()), detection={}, selection={})
    T = {k: np.array([f(s) for s in sc]) for k, f in STATS.items()}
    uq = np.unique(q)
    for k, t in T.items():
        d = dict(auroc=auroc(t[pos], t[~pos]))
        bs, miss, fa = [], [], []
        for b in range(1000):
            pick = rng.choice(uq, len(uq))
            m = np.isin(q, pick)  # approximate cluster bootstrap (set membership) for speed
            bs.append(auroc(t[m & pos], t[m & ~pos]))
        d["auroc_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        for sp in range(50):
            half = np.isin(q, rng.choice(uq, len(uq) // 2, replace=False))
            tau = conformal_q(t[half & ~pos], a.alpha)
            fa.append(float((t[~half & ~pos] > tau).mean())); miss.append(float((t[~half & pos] <= tau).mean()))
        d.update(false_alarm=float(np.mean(fa)), miss_at_alpha=float(np.mean(miss)), miss_sd=float(np.std(miss)))
        res["detection"][k] = d
        print(f"{k:12s} AUROC {100*d['auroc']:.2f} [{100*d['auroc_ci95'][0]:.2f},{100*d['auroc_ci95'][1]:.2f}]  FA {d['false_alarm']:.3f}  miss@{a.alpha} {d['miss_at_alpha']:.3f}", flush=True)
    # paired AUROC differences vs max
    base = T["max"]
    for k, t in T.items():
        if k == "max":
            continue
        ds = []
        for b in range(500):
            m = np.isin(q, rng.choice(uq, len(uq)))
            ds.append(auroc(t[m & pos], t[m & ~pos]) - auroc(base[m & pos], base[m & ~pos]))
        res["detection"][k]["delta_auroc_vs_max_ci95"] = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
    # selection inside positive scans
    P = [(s, np.array(x["labels"], bool)) for s, x in zip(sc, S) if x["pos"] and x["labels"] and any(x["labels"])]
    for mode in ("indep", "smooth", "nms"):
        h1 = [lab[select(s, 1, mode)[0]] for s, lab in P]
        k3 = [lab[select(s, 3, mode)].any() for s, lab in P]
        prec3 = [lab[select(s, 3, mode)].mean() for s, lab in P]
        res["selection"][mode] = dict(n=len(P), hit_at_1=float(np.mean(h1)), any_hit_at_3=float(np.mean(k3)), precision_at_3=float(np.mean(prec3)))
        print("select", mode, res["selection"][mode], flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/setsel.json", "w"), indent=1)


if __name__ == "__main__":
    main()
