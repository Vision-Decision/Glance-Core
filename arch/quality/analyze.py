"""Exp 3 analysis (CPU): box quality read analytically off the edge/centre distributions vs the learned IoU head (kappa_hat)
and vs the AR-digits + trained IoU head baseline (outputs/ablation_v3/ar_iou), on the controlled-study checkpoint
(ours_neg25) and on Glance v1.1.

  python arch/quality/analyze.py --dumps DIR (with ours_neg25/ and v1.1/ *_scalar.npz from arch/quality/dump.py)
         --out outputs/arch/quality

Test set and protocols are exactly ablation/twoknob_v3.py's: gRefCOCO val merged, 10,834 single-target present +
8,905 no-target items; P1 = image-half cross-fitted Platt on the test set; P2 = Platt fitted on C (gRefCOCO testA+B,
image-disjoint) with the presence prior shifted to the known test rate. Rule: answer iff (cR-cW+cH) eta + cW eta kappa > cH.
CIs: 1000x image bootstrap, paired for differences.
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ablation"))
from twoknob_v2 import platt_fit, logit, cost_vec, boot_ci  # noqa: E402
from selective_v2 import aurc, risk_at_cov  # noqa: E402

CW = [0.0, 0.5, 1.0, 2.0, 5.0]
B = 1000


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def auroc(s, y):
    r = rankdata(s); y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p, y, n_bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(n_bins) if (b == k).any()))


def boot_idx(img, B=B, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(img, return_inverse=True)
    mem = [[] for _ in range(len(u))]
    for i, g in enumerate(inv):
        mem[g].append(i)
    mem = [np.array(m) for m in mem]
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        yield np.concatenate([mem[p] for p in pick])


def auc_with_ci(scores, labels, img, base=None):
    """scores: dict name -> (s, y); returns auroc + CI, and paired diff vs base (name) with CI."""
    out = {k: dict(auroc=auroc(s, y)) for k, (s, y) in scores.items()}
    bs = {k: [] for k in scores}
    for idx in boot_idx(img):
        for k, (s, y) in scores.items():
            yy = y[idx]
            if 0 < yy.sum() < len(yy):
                bs[k].append(auroc(s[idx], yy))
            else:
                bs[k].append(np.nan)
    for k in scores:
        v = np.array(bs[k]); out[k]["ci95"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    if base:
        for k in scores:
            if k == base:
                continue
            d = np.array(bs[k]) - np.array(bs[base])
            out[k][f"delta_vs_{base}"] = out[k]["auroc"] - out[base]["auroc"]
            out[k][f"delta_vs_{base}_ci95"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    return out


def crossfit_lr(X, y, img, mask):
    """2-fold image-half cross-fitted logistic regression on features X (n,d) (for 'does analytic add to kappa_hat')."""
    from scipy.optimize import minimize
    half = img % 2 == 0
    out = np.zeros(len(y))
    Xb = np.c_[X, np.ones(len(X))]
    for tr, te in ((half, ~half), (~half, half)):
        t = tr & mask
        def f(w):
            z = Xb[t] @ w; p = sig(z)
            ll = -(y[t] * np.log(p + 1e-12) + (1 - y[t]) * np.log(1 - p + 1e-12)).mean()
            g = Xb[t].T @ (p - y[t]) / t.sum()
            return ll + 1e-6 * (w @ w), g + 2e-6 * w
        w = minimize(f, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B").x
        out[te] = Xb[te] @ w
    return out


def apply(w, x):
    return sig(w[0] * x + w[1])


def crossfit(fx, target, mask, img):
    half = img % 2 == 0
    out = np.zeros(len(fx))
    for tr, te in ((half, ~half), (~half, half)):
        w = platt_fit(fx[tr & mask], target[tr & mask].astype(float))
        out[te] = apply(w, fx[te])
    return out


def shift(p, a, b):
    return sig(logit(p) + logit(b) - logit(a))


def two_knob(eta, kap, cW, cH=1.0, cR=1.0):
    return (cR - cW + cH) * eta + cW * eta * kap > cH


def keep_idx(present, iou):
    return np.where((~present) | (iou >= 0))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--items", default="outputs/ablation_v3/ar_iou/items.npz")
    ap.add_argument("--rows", default="outputs/ablation/ours_neg25/rows.json")
    ap.add_argument("--out", default="outputs/arch/quality")
    a = ap.parse_args()
    I = dict(np.load(a.items))
    res = dict(sources=dict(dumps=a.dumps, items=a.items, rows=a.rows,
                            hf_full_dumps="Vision-Decision/glance-train-data:arch/quality_dumps/"))
    # ---------------------------------------------------------------- controlled-study checkpoint: align with items.npz
    T = dict(np.load(f"{a.dumps}/ours_neg25/grefcoco_val_merged_scalar.npz"))
    C = dict(np.load(f"{a.dumps}/ours_neg25/arcal_gtest_scalar.npz"))
    kt = keep_idx(T["present"], T["iou"]); kc = keep_idx(C["present"], C["iou"])
    chk = dict(n_test=len(kt), n_items=len(I["t_y"]), n_cal=len(kc), n_items_cal=len(I["c_y"]))
    assert len(kt) == len(I["t_y"]) and len(kc) == len(I["c_y"]), chk
    chk["test_max_abs_dkappa"] = float(np.abs(T["p_iou50"][kt] - I["t_k"]).max())
    chk["test_max_abs_deta"] = float(np.abs(T["p_present"][kt] - I["t_g"]).max())
    chk["test_good_agree"] = float(np.mean((T["present"][kt] & (T["iou"][kt] >= .5)) == I["t_good_g"]))
    chk["cal_max_abs_dkappa"] = float(np.abs(C["p_iou50"][kc] - I["c_k"]).max())
    chk["cal_good_agree"] = float(np.mean((C["present"][kc] & (C["iou"][kc] >= .5)) == I["c_good_g"]))
    res["alignment"] = chk
    print("[align]", chk, flush=True)
    y, img = I["t_y"], I["t_img"]
    P = y
    feats = ["E_edge", "P50_edge", "E_mix", "P50_mix", "P50_mixfull", "E_mix_T0.5", "P50_mix_T0.5", "P50_mix_T2",
             "P50_edge_T0.5", "P50_edge_T2", "top1", "top_mass"]
    t = {f: T[f][kt] for f in feats}
    t["neg_H_center"] = -T["H_center"][kt]; t["neg_H_edges"] = -T["H_edges"][kt]
    c = {f: C[f][kc] for f in feats}
    good_g, good_a = I["t_good_g"], I["t_good_a"]
    # analytic + kappa combined (cross-fitted LR on logits; present items only)
    Xc = np.c_[logit(np.clip(I["t_k"], 1e-6, 1 - 1e-6)), logit(np.clip(t["P50_mix"], 1e-4, 1 - 1e-4)),
               logit(np.clip(t["E_mix"], 1e-4, 1 - 1e-4)), t["neg_H_center"]]
    combo = crossfit_lr(Xc, good_g.astype(float), img, P)
    # ---------------------------------------------------------------- box-quality diagnostics on present items
    S = {"kappa_hat (learned IoU head)": (I["t_k"][P], good_g[P]),
         "AR+IoU head (heldout_full)": (sig(I["t_head_heldout_full"])[P], good_a[P])}
    for f, v in t.items():
        S[f"analytic:{f}"] = (v[P], good_g[P])
    S["combo: kappa_hat + analytic (LR, cross-fit)"] = (combo[P], good_g[P])
    sp = f"{a.dumps}/ours_neg25/grefcoco_val_merged_stats.npz"
    if os.path.exists(sp):  # option (b): a cross-fitted linear readout of distribution statistics (learned, but tiny)
        St = dict(np.load(sp))
        F = St["F"][kt].astype(float); F = (F - F[P].mean(0)) / (F[P].std(0) + 1e-6)
        S["analytic:agree_w (centre-weighted IoU of top-8 point boxes)"] = (St["agree_w"][kt][P], good_g[P])
        S["learned-on-stats: LR(edge/centre stats), cross-fit"] = (crossfit_lr(F, good_g.astype(float), img, P)[P], good_g[P])
        S["learned-on-stats + kappa_hat, cross-fit"] = (crossfit_lr(np.c_[F, Xc[:, :1]], good_g.astype(float), img, P)[P], good_g[P])
    D = auc_with_ci(S, None, img[P], base="kappa_hat (learned IoU head)")
    D2 = auc_with_ci({k: S[k] for k in ["AR+IoU head (heldout_full)", "analytic:P50_mix", "analytic:E_mix", "analytic:P50_mixfull"]},
                     None, img[P], base="AR+IoU head (heldout_full)")
    for k in D2:
        if k != "AR+IoU head (heldout_full)":
            D[k]["delta_vs_AR+IoU head"] = D2[k]["delta_vs_AR+IoU head (heldout_full)"]
            D[k]["delta_vs_AR+IoU head_ci95"] = D2[k]["delta_vs_AR+IoU head (heldout_full)_ci95"]
    for k, (s, g) in S.items():
        D[k].update(risk_at_80cov=risk_at_cov(s, g, 0.8), aurc_pos=aurc(s, g), base_error=float(1 - g.mean()))
        if k.startswith("analytic:P50") or k.startswith("analytic:E_") or k.startswith("kappa") or k.startswith("AR+"):
            D[k]["ece_raw"] = ece(s, g)
    res["box_quality_present_test"] = D
    # ---------------------------------------------------------------- two-knob cost with the analytic kappa
    cal = {}
    pi_t, pi_c = float(y.mean()), float(I["c_y"].mean())
    fe_g, fe_gc = logit(I["t_g"]), logit(I["c_g"])
    eta_P1 = crossfit(fe_g, y, np.ones_like(y), img)
    we = platt_fit(fe_gc, I["c_y"].astype(float)); eta_P2 = shift(apply(we, fe_g), pi_c, pi_t)
    kinds = {"kappa_hat": (I["t_k"], I["c_k"]), "analytic_P50_mix": (t["P50_mix"], c["P50_mix"]),
             "analytic_E_mix": (t["E_mix"], c["E_mix"]), "analytic_P50_mixfull": (t["P50_mixfull"], c["P50_mixfull"])}
    for k, (kt_, kc_) in kinds.items():
        fk, fkc = logit(np.clip(kt_, 1e-4, 1 - 1e-4)), logit(np.clip(kc_, 1e-4, 1 - 1e-4))
        k1 = crossfit(fk, good_g, y, img)
        wk = platt_fit(fkc[I["c_y"]], I["c_good_g"][I["c_y"]].astype(float)); k2 = apply(wk, fk)
        cal[f"P1:{k}"] = (eta_P1, k1); cal[f"P2:{k}"] = (eta_P2, k2)
        cal[f"raw:{k}"] = (I["t_g"], kt_)                          # no calibration at all (plug-in on raw outputs)
    # AR + head under the same protocols (from items.npz)
    fa, fac = I["t_ar_margin"], I["c_ar_margin"]
    ea1 = crossfit(fa, y, np.ones_like(y), img)
    wa = platt_fit(fac, I["c_y"].astype(float)); ea2 = shift(apply(wa, fa), pi_c, pi_t)
    fh, fhc = I["t_head_heldout_full"], I["c_head_heldout_full"]
    cal["P1:AR+head"] = (ea1, crossfit(fh, good_a, y, img))
    wh = platt_fit(fhc[I["c_y"]], I["c_good_a"][I["c_y"]].astype(float)); cal["P2:AR+head"] = (ea2, apply(wh, fh))
    costs = {}
    for cW in CW:
        row, vec = {}, {}
        for k, (eta, kap) in cal.items():
            g = good_a if "AR" in k else good_g
            cv = cost_vec(two_knob(eta, kap, cW), dict(y=y, good=g), (1.0, 1.0, cW))
            vec[k] = cv
            row[k] = dict(cost=float(cv.mean()), ci95=boot_ci(cv, img), coverage=float(two_knob(eta, kap, cW).mean()))
        for pr in ("P1", "P2", "raw"):
            base = vec[f"{pr}:kappa_hat"]
            for k in vec:
                if k.startswith(pr + ":") and k != f"{pr}:kappa_hat":
                    d = vec[k] - base
                    row[f"delta[{k} - {pr}:kappa_hat]"] = dict(mean=float(d.mean()), ci95=boot_ci(d, img, seed=2))
            if pr != "raw":
                for k in ("analytic_P50_mix", "analytic_E_mix"):
                    d = vec[f"{pr}:{k}"] - vec[f"{pr}:AR+head"]
                    row[f"delta[{pr}:{k} - {pr}:AR+head]"] = dict(mean=float(d.mean()), ci95=boot_ci(d, img, seed=3))
        costs[f"cW={cW}"] = row
    res["two_knob_cost_x1"] = costs
    # ---------------------------------------------------------------- v1.1 (release-family model), gRefCOCO val merged
    V = dict(np.load(f"{a.dumps}/v1.1/grefcoco_val_merged_scalar.npz"))
    pm = V["present"] & (V["iou"] >= 0)
    gv = V["iou"][pm] >= .5
    imgv = np.unique(V["image_key"], return_inverse=True)[1][pm]
    SV = {"kappa_hat": (V["p_iou50"][pm], gv)}
    for f in ["E_edge", "P50_edge", "E_mix", "P50_mix", "P50_mixfull", "top1"]:
        SV[f"analytic:{f}"] = (V[f][pm], gv)
    DV = auc_with_ci(SV, None, imgv, base="kappa_hat")
    for k, (s, g) in SV.items():
        DV[k].update(ece_raw=ece(s, g) if not k.endswith("top1") else None, risk_at_80cov=risk_at_cov(s, g, 0.8))
    res["v1.1_box_quality_present"] = dict(n=int(pm.sum()), acc50=float(gv.mean()), scores=DV)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
    for k, v in D.items():
        print(f"{k:50s} AUROC {100*v['auroc']:.1f} [{100*v['ci95'][0]:.1f},{100*v['ci95'][1]:.1f}]"
              + (f"  d_vs_kappa {100*v.get('delta_vs_kappa_hat (learned IoU head)', 0):+.1f}" if "delta_vs_kappa_hat (learned IoU head)" in v else "")
              + (f"  ECE {100*v['ece_raw']:.2f}" if v.get("ece_raw") is not None else ""))
    for k, v in DV.items():
        print("v1.1", k, round(100 * v["auroc"], 1), v["ci95"], v.get("ece_raw"))
    for cW in ("cW=1.0", "cW=5.0"):
        print(cW, {k: round(100 * v["cost"], 2) for k, v in costs[cW].items() if not k.startswith("delta")})


if __name__ == "__main__":
    main()
