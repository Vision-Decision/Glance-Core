"""Tab. 2 (Prop. 1 two-knob cost) with an AR grounder that has its own trained IoU head, under two calibration
protocols.  CPU only.

  build:    python ablation/twoknob_v3.py build --raw DIR      (DIR = unpacked v3_small.tgz from ablation/v3_ar_iou.sh)
            -> outputs/ablation_v3/ar_iou/items.npz (compact per-item arrays, test + calibration split)
  analyse:  python ablation/twoknob_v3.py [--out outputs/ablation_v3/ar_iou]  -> results.json

Test items: gRefCOCO val merged, 10,834 single-target present + 8,905 no-target (= twoknob_v2.py). Glance = ours_neg25
rows (outputs/ablation/ours_neg25/rows.json); AR-digits = ar_digits_neg25 constrained rows (rows_c.json; the VM re-decode
must reproduce them, checked in build). AR + IoU head: AR-digits' presence (margin to "none") with box quality from the
trained head of ablation/ar_iou_head.py (frozen AR model; main variant heldout_full).
Protocols
  P1 cross-fitted (as twoknob_v2.py): Platt on test image halves, each half scored with the other half's fit.
  P2 label-free: Platt fitted only on C = gRefCOCO testA+testB (single-target + no-target, images disjoint from test and
     from v0 training); presence logit shifted from C's present rate to the test present rate, the latter either
     known (aggregate only, 'known_prior') or estimated without labels by EM on the test scores (Saerens et al. 2002,
     'em_prior'). kappa: no shift.
Rule: Prop. 1 plug-in, answer iff (c_R - c_W + c_H) eta + c_W eta kappa > c_H;  c_H = c_R = 1, c_W in {0,.5,1,2,5}.
95% CIs: image bootstrap (1000) of per-item cost; paired for differences.
Box-quality diagnostics on the 10,834 present test items: AUROC of kappa for IoU>=.5 vs <.5, ECE (15 bins; raw head and
after each protocol's Platt), positive-only selective risk at 80% coverage, positive-only AURC.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twoknob_v2 import load, platt_fit, logit, cost_vec, boot_ci  # noqa: E402
from selective_v2 import n_box_tokens, aurc, risk_at_cov  # noqa: E402

OUT = "outputs/ablation_v3/ar_iou"
CW = [0.0, 0.5, 1.0, 2.0, 5.0]
HEADS = ["heldout_full", "heldout_scalar", "insample_full"]
SRC = "outputs/ablation"
PREP = "data/prep/eval_grefcoco_val_merged.jsonl"


# ----------------------------------------------------------------------------------------------- build
def keep_idx(rows):
    return [k for k, r in enumerate(rows) if (not r["present"]) or ("iou" in r)]


def ar_arrays(R, run="ar_digits_neg25"):
    pn = np.array([r["p_none"] for r in R]); lpb = np.array([r["lp_box"] for r in R])
    L = np.array([n_box_tokens(run, r["box"]) if r.get("box") else 4 for r in R])
    pmc = np.array([r.get("p_maxcoord_c", r["p_maxcoord"]) for r in R])
    lpf = np.log(np.clip(pmc, 1e-30, 1))
    return dict(ar_margin=lpf - np.log(np.clip(pn, 1e-30, 1)), ar_kap=np.exp((lpb - lpf) / np.maximum(L - 1, 1)),
                ar_pshat=np.exp(lpb))


def build(raw, out):
    hs = np.load(f"{raw}/head_scores.npz")
    res, chk = {}, {}
    # test: rows from the ablation runs, head scores aligned by row order (re-decode must match rows_c)
    old = json.load(open(f"{SRC}/ar_digits_neg25/rows_c.json"))["grefcoco_val_merged"]
    new = json.load(open(f"{raw}/feats/rows_grefcoco_val_merged.json"))
    assert len(old) == len(new)
    same_box = np.mean([o["box"] == n["box"] for o, n in zip(old, new)])
    dlp = np.abs(np.array([o["lp_box"] - n["lp_box"] for o, n in zip(old, new)]))
    dpn = np.abs(np.array([o["p_none"] - n["p_none"] for o, n in zip(old, new)]))
    chk["test_redecode_vs_rows_c"] = dict(same_box=float(same_box), max_abs_dlp_box=float(dlp.max()),
                                          median_abs_dlp_box=float(np.median(dlp)), max_abs_dp_none=float(dpn.max()))
    kt = keep_idx(old)
    dg, da = load(SRC, "ours_neg25", PREP), load(SRC, "ar_digits_neg25", PREP)
    assert np.array_equal(dg["img"], da["img"]) and np.array_equal(dg["y"], da["y"])
    res.update(t_y=dg["y"], t_good_g=dg["good"], t_good_a=da["good"], t_img=dg["img"], t_g=dg["eta"], t_k=dg["kap"],
               t_ar_margin=da["margin"], t_ar_kap=da["kap"])
    R = [old[k] for k in kt]
    res["t_ar_pshat"] = ar_arrays(R)["ar_pshat"]
    for h in HEADS:
        res[f"t_head_{h}"] = hs[f"{h}__grefcoco_val_merged"][kt]
    # calibration split C (all items are single-target-with-box or no-target by construction)
    ra = json.load(open(f"{raw}/feats/rows_arcal_gtest.json"))
    rg = json.load(open(f"{raw}/ours_rows_arcal.json"))["arcal_gtest"]
    assert len(ra) == len(rg)
    assert all(a["present"] == g["present"] for a, g in zip(ra, rg))
    kc = keep_idx(ra)
    ra = [ra[k] for k in kc]; rg = [rg[k] for k in kc]
    y = np.array([r["present"] for r in ra])
    A = ar_arrays(ra)
    res.update(c_y=y, c_img=np.array([r["image_id"] for r in ra]),
               c_good_a=y & (np.array([r.get("iou", 0) for r in ra]) >= .5),
               c_good_g=y & (np.array([r.get("iou", 0) for r in rg]) >= .5),
               c_g=np.array([r["p_present"] for r in rg]), c_k=np.array([r["p_iou50"] for r in rg]),
               c_ar_margin=A["ar_margin"], c_ar_kap=A["ar_kap"])
    for h in HEADS:
        res[f"c_head_{h}"] = hs[f"{h}__arcal_gtest"][kc]
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(f"{out}/items.npz", **res)
    chk["n_test"] = int(len(res["t_y"])); chk["n_cal"] = int(len(y)); chk["cal_absent"] = int((~y).sum())
    chk["cal_acc50_ar"] = float(res["c_good_a"][y].mean()); chk["cal_acc50_gate"] = float(res["c_good_g"][y].mean())
    json.dump(chk, open(f"{out}/build_check.json", "w"), indent=1)
    print(json.dumps(chk, indent=1))


# ----------------------------------------------------------------------------------------------- analysis
def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def ece(p, y, n_bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def auroc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def apply(w, x):
    return sig(w[0] * x + w[1])


def crossfit(fx, target, mask, img):
    """2-fold image-half cross-fitted Platt (twoknob_v2.platt_crossfit); fit on items in mask."""
    half = img % 2 == 0
    out = np.zeros(len(fx))
    for tr, te in ((half, ~half), (~half, half)):
        w = platt_fit(fx[tr & mask], target[tr & mask].astype(float))
        out[te] = apply(w, fx[te])
    return out


def em_prior(p, pi_src, iters=200):
    """Saerens et al. (2002): estimate the test prior of 'present' from calibrated source posteriors p (no labels)."""
    pi = pi_src
    for _ in range(iters):
        a = pi / pi_src * p; b = (1 - pi) / (1 - pi_src) * (1 - p)
        pi_new = float((a / (a + b)).mean())
        if abs(pi_new - pi) < 1e-9:
            break
        pi = pi_new
    return pi


def shift(p, pi_src, pi_tgt):
    return sig(logit(p) + logit(pi_tgt) - logit(pi_src))


def feats(I, pre):
    """Platt input features per method (as twoknob_v2: logit for Glance, margin / log-conf for AR, logit for heads)."""
    f = {"glance": (logit(I[f"{pre}_g"]), logit(I[f"{pre}_k"])),
         "ar": (I[f"{pre}_ar_margin"], np.log(np.clip(I[f"{pre}_ar_kap"], 1e-12, 1)))}
    for h in HEADS:
        f[f"ar+{h}"] = (I[f"{pre}_ar_margin"], I[f"{pre}_head_{h}"])
    return f


def good(I, pre, m):
    return I[f"{pre}_good_g"] if m == "glance" else I[f"{pre}_good_a"]


def calibrate(I):
    """Returns {protocol: {method: (eta, kappa)}} on test items."""
    ft, fc = feats(I, "t"), feats(I, "c")
    y, img = I["t_y"], I["t_img"]
    out = {"P1_crossfit": {}, "P2_known_prior": {}, "P2_em_prior": {}}
    pi_t_true = float(y.mean()); pi_c = float(I["c_y"].mean())
    priors = {}
    for m in ft:
        fe, fk = ft[m]
        out["P1_crossfit"][m] = (crossfit(fe, y, np.ones_like(y), img), crossfit(fk, good(I, "t", m), y, img))
        ce, ck = fc[m]
        we = platt_fit(ce, I["c_y"].astype(float))
        wk = platt_fit(ck[I["c_y"]], good(I, "c", m)[I["c_y"]].astype(float))
        eta_c, kap_c = apply(we, fe), apply(wk, fk)
        pi_em = em_prior(eta_c, pi_c)
        priors[m] = dict(pi_cal=pi_c, pi_test_true=pi_t_true, pi_test_em=pi_em)
        out["P2_known_prior"][m] = (shift(eta_c, pi_c, pi_t_true), kap_c)
        out["P2_em_prior"][m] = (shift(eta_c, pi_c, pi_em), kap_c)
    return out, priors


def two_knob(eta, kap, cW, cH=1.0, cR=1.0):
    return (cR - cW + cH) * eta + cW * eta * kap > cH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="analyse")
    ap.add_argument("--raw", default="")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.cmd == "build":
        return build(a.raw, a.out)
    I = dict(np.load(f"{a.out}/items.npz"))
    y, img = I["t_y"], I["t_img"]
    res = dict(n_test=int(len(y)), n_present=int(y.sum()), n_cal=int(len(I["c_y"])))
    cal, res["priors"] = calibrate(I)
    # ---- box-quality diagnostics on present test items
    P = y
    diag = {}
    raw = {"glance_kappa_hat": (I["t_k"], I["t_good_g"]), "ar_logprob_P(s_hat)": (I["t_ar_pshat"], I["t_good_a"]),
           "ar_lenorm_conf": (I["t_ar_kap"], I["t_good_a"])}
    for h in HEADS:
        raw[f"ar_head_{h}"] = (sig(I[f"t_head_{h}"]), I["t_good_a"])
    for k, (s, g) in raw.items():
        s, g = s[P], g[P]
        d = dict(auroc=auroc(s, g), risk_at_80cov=risk_at_cov(s, g, 0.8), aurc_pos=aurc(s, g), base_error=float(1 - g.mean()))
        if k.startswith("glance") or k.startswith("ar_head"):
            d["ece_raw"] = ece(s, g)
        m = "glance" if k.startswith("glance") else ("ar+" + k[len("ar_head_"):] if k.startswith("ar_head") else
                                                     ("ar" if k == "ar_lenorm_conf" else None))
        if m:
            for pr in cal:
                d[f"ece_{pr}"] = ece(cal[pr][m][1][P], g)
        diag[k] = d
    res["box_quality_present_test"] = diag
    # ---- cost table
    tab = {}
    for pr, C in cal.items():
        tp = {}
        for cW in CW:
            row, acc = {}, {}
            for m, (eta, kap) in C.items():
                g = I["t_good_g"] if m == "glance" else I["t_good_a"]
                d = dict(y=y, good=g)
                for rule, ac in (("two_knob", two_knob(eta, kap, cW)), ("presence", eta > 0.5)):
                    cv = cost_vec(ac, d, (1.0, 1.0, cW))
                    acc[(m, rule)] = cv
                    row[f"{m}:{rule}"] = dict(cost=float(cv.mean()), ci95=boot_ci(cv, img), coverage=float(ac.mean()))
            base = acc[("glance", "two_knob")]
            for m in C:
                if m == "glance":
                    continue
                diff = base - acc[(m, "two_knob")]
                row[f"delta_glance_minus_{m}"] = dict(mean=float(diff.mean()), ci95=boot_ci(diff, img, seed=2),
                                                      rel=float(diff.mean() / acc[(m, "two_knob")].mean()))
                if m != "ar":
                    dd = acc[(m, "two_knob")] - acc[("ar", "two_knob")]
                    row[f"delta_{m}_minus_ar"] = dict(mean=float(dd.mean()), ci95=boot_ci(dd, img, seed=3))
            tp[f"cW={cW}"] = row
        tab[pr] = tp
    res["costs_cH=cR=1"] = tab
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
    for pr, tp in tab.items():
        print(pr)
        for k, row in tp.items():
            print(" ", k, " ".join(f"{m}={row[f'{m}:two_knob']['cost']*100:.2f}" for m in cal[pr]),
                  "| d(G-AR+head)=%.2f [%.2f,%.2f]" % tuple(100 * np.array([row["delta_glance_minus_ar+heldout_full"]["mean"],
                                                                             *row["delta_glance_minus_ar+heldout_full"]["ci95"]])))
    print(json.dumps(diag, indent=1))
    print(json.dumps(res["priors"], indent=1))


if __name__ == "__main__":
    main()
