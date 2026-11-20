"""Gen-v1 vs Glance v1 (CPU; numpy + scipy).

python gen_v1/analyze.py --v1 DIR_v1_rows --gen DIR_gen_rows [--venus_gen DIR --venus_v1 DIR] --out outputs/gen_v1

Inputs: Glance v1 rows = HF Vision-Decision/glance-train-ckpts-legacy sprint_abs/rows_v1/<set>.jsonl.gz (scripts/sprint_abs/dump.py);
Gen-v1 rows = gen_v1/dump.py (rows_<set>.jsonl.gz, head_scores.npz from gen_v1/iou_head.py). Items are joined on
(image, expression, presence, GT box) so both systems are scored on identical questions.

Read-outs of Gen-v1: presence dial 1 - q(none) at the first answer token and the greedy fixed point (argmax = "none");
box = grammar-constrained greedy decode with the first token forced to a coordinate (standard Acc@0.5);
kappa = post-hoc IoU head (held-out H), plus the token-probability baselines.
Two-knob cost: ablation/twoknob_v3.py protocols (P1 cross-fitted Platt on test image halves; P2 Platt on C = gRefCOCO
testA+testB single/no-target + prior shift to known / EM-estimated test rate) and the box-accuracy-equalised variant
(ablation/twoknob_v3_boxeq.py). 95% CIs: image bootstrap (1000), paired for differences.
"""
import argparse, collections, gzip, json, os, re, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ablation"))
import twoknob_v3 as T  # noqa: E402
from twoknob_v2 import cost_vec, boot_ci  # noqa: E402
from selective_v2 import n_box_tokens  # noqa: E402

HEADS = ["heldout_full", "heldout_scalar"]
T.HEADS = HEADS
SPLITS = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocoplus_testA", "refcocoplus_testB",
          "refcocog_val", "refcocog_test"]


def rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def key(r):
    return (r["img"], r["text"], bool(r["present"]), json.dumps(r.get("gt_box")))


def join(a, b):
    """-> index arrays ia, ib aligning rows of a and b on key (with multiplicity)."""
    pos = collections.defaultdict(list)
    for j, r in enumerate(b):
        pos[key(r)].append(j)
    ia, ib = [], []
    for i, r in enumerate(a):
        k = key(r)
        if pos[k]:
            ia.append(i); ib.append(pos[k].pop(0))
    return np.array(ia), np.array(ib)


def img_int(r):
    m = re.search(r"(\d{6,})", os.path.basename(r["img"]))
    return int(m.group(1)) if m else abs(hash(r["img"])) % (10 ** 9)


def sig(z):
    return 1 / (1 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))


def gen_arrays(R):
    pn = np.array([r["p_none"] for r in R]); lpb = np.array([r["lp_box"] for r in R])
    L = np.array([n_box_tokens("digits", r["box"]) if r.get("box") else 4 for r in R])
    pmc = np.array([r["p_maxcoord_c"] for r in R])
    lpf = np.log(np.clip(pmc, 1e-30, 1))
    return dict(margin=lpf - np.log(np.clip(pn, 1e-30, 1)), kap=np.exp((lpb - lpf) / np.maximum(L - 1, 1)),
                pshat=np.exp(lpb), p_none=pn, greedy_none=np.array([r["greedy_none"] for r in R]),
                greedy_none_c=np.array([r["greedy_none_c"] for r in R]))


def boot(fn, groups, B=1000, seed=0):
    """image-cluster bootstrap of a scalar statistic fn(idx) -> 95% CI"""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True)
    members = [[] for _ in range(len(u))]
    for i, g in enumerate(inv):
        members[g].append(i)
    members = [np.array(m) for m in members]
    vals = []
    for _ in range(B):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(u), len(u))])
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def stat(fn, groups, B=1000, seed=0):
    return dict(value=float(fn(np.arange(len(groups)))), ci95=boot(fn, groups, B, seed))


# ----------------------------------------------------------------------------------------------- RefCOCO
def refcoco(v1, gen, merged_gen):
    out = {}
    for sp in SPLITS:
        a = rows(f"{v1}/{sp}.jsonl.gz")
        if sp == "refcoco_val":   # single-target part of the merged gRefCOCO val decode (same images / expressions)
            b = [r for r in merged_gen if r["present"] and "iou" in r]
        elif os.path.exists(f"{gen}/rows_{sp}.jsonl.gz"):
            b = rows(f"{gen}/rows_{sp}.jsonl.gz")
        else:
            continue
        ia, ib = join(a, b)
        A = [a[i] for i in ia]; Bq = [b[i] for i in ib]
        ga = np.array([r["iou"] >= .5 for r in A], float); gb = np.array([r["iou"] >= .5 for r in Bq], float)
        g = np.array([img_int(r) for r in A])
        out[sp] = dict(n=len(A), n_v1_rows=len(a), n_gen_rows=len(b),
                       glance_v1_acc50=stat(lambda i: 100 * ga[i].mean(), g),
                       gen_v1_acc50=stat(lambda i: 100 * gb[i].mean(), g),
                       delta_glance_minus_gen=stat(lambda i: 100 * (ga[i] - gb[i]).mean(), g, seed=1),
                       gen_parse_ok=float(np.mean([r["parse"] == "box" for r in Bq])))
    done = [s for s in SPLITS if s in out]
    if len(done) == len(SPLITS):
        out["avg8"] = dict(glance_v1=float(np.mean([out[s]["glance_v1_acc50"]["value"] for s in SPLITS])),
                           gen_v1=float(np.mean([out[s]["gen_v1_acc50"]["value"] for s in SPLITS])))
    return out


# ----------------------------------------------------------------------------------------------- gRefCOCO val merged
def grec(v1rows, genrows, hs):
    ia, ib = join(v1rows, genrows)
    assert len(ia) == len(v1rows) == len(genrows), (len(ia), len(v1rows), len(genrows))
    A = [v1rows[i] for i in ia]; Bq = [genrows[i] for i in ib]
    y = np.array([r["present"] for r in A])
    g = np.array([img_int(r) for r in A])
    G = gen_arrays(Bq)
    pg = sig([r["p_logit"] for r in A])
    out = dict(n=len(A), n_present=int(y.sum()), n_absent=int((~y).sum()))
    ab = ~y
    pres_single = np.array([r["present"] and "iou" in r for r in A])
    multi = y & ~pres_single
    # absence
    out["N_acc"] = dict(
        glance_gate_0_5=stat(lambda i: 100 * (pg[i][ab[i]] < .5).mean(), g),
        gen_greedy=stat(lambda i: 100 * G["greedy_none"][i][ab[i]].mean(), g),
        gen_greedy_grammar=stat(lambda i: 100 * G["greedy_none_c"][i][ab[i]].mean(), g),
        gen_pnone_0_5=stat(lambda i: 100 * (G["p_none"][i][ab[i]] > .5).mean(), g),
        delta_glance_minus_gen_greedy=stat(lambda i: 100 * ((pg[i][ab[i]] < .5).mean() - G["greedy_none"][i][ab[i]].mean()), g, seed=1))
    out["refusal_single_target"] = dict(glance=float((pg[pres_single] < .5).mean() * 100),
                                        gen_greedy=float(G["greedy_none"][pres_single].mean() * 100))
    out["refusal_multi_target"] = dict(glance=float((pg[multi] < .5).mean() * 100),
                                       gen_greedy=float(G["greedy_none"][multi].mean() * 100))
    # presence AUROC / ECE (multi-target counts as present); raw scores and cross-fitted Platt
    eta_gen_raw = 1 - G["p_none"]
    half = g % 2 == 0
    ones = np.ones_like(y)
    eta_g_p1 = T.crossfit(T.logit(pg), y, ones, g)
    eta_a_p1 = T.crossfit(G["margin"], y, ones, g)
    out["presence"] = dict(
        auroc_glance=stat(lambda i: 100 * T.auroc(pg[i], y[i]), g),
        auroc_gen_pnone=stat(lambda i: 100 * T.auroc(eta_gen_raw[i], y[i]), g),
        auroc_gen_margin=stat(lambda i: 100 * T.auroc(G["margin"][i], y[i]), g),
        delta_auroc_glance_minus_gen_pnone=stat(lambda i: 100 * (T.auroc(pg[i], y[i]) - T.auroc(eta_gen_raw[i], y[i])), g, seed=1),
        ece_raw_glance=stat(lambda i: 100 * T.ece(pg[i], y[i]), g),
        ece_raw_gen_1_minus_pnone=stat(lambda i: 100 * T.ece(eta_gen_raw[i], y[i]), g),
        ece_crossfit_platt_glance=stat(lambda i: 100 * T.ece(eta_g_p1[i], y[i]), g),
        ece_crossfit_platt_gen_margin=stat(lambda i: 100 * T.ece(eta_a_p1[i], y[i]), g))
    # box (forced) and kappa on single-target present items
    P = pres_single
    gp = g[P]
    good_g = np.array([A[k]["iou"] >= .5 for k in np.where(P)[0]])
    good_a = np.array([Bq[k]["iou"] >= .5 for k in np.where(P)[0]])
    kg = sig([A[k]["iou_logit"] for k in np.where(P)[0]])
    head = {h: sig(hs[f"{h}__grefcoco_val_merged"][ib][P]) for h in HEADS}
    out["box_acc50_single_target"] = dict(glance=stat(lambda i: 100 * good_g[i].mean(), gp),
                                          gen=stat(lambda i: 100 * good_a[i].mean(), gp),
                                          delta=stat(lambda i: 100 * (good_g[i].mean() - good_a[i].mean()), gp, seed=1))
    k = dict(auroc_glance_kappa_hat=stat(lambda i: 100 * T.auroc(kg[i], good_g[i]), gp),
             ece_raw_glance_kappa_hat=stat(lambda i: 100 * T.ece(kg[i], good_g[i]), gp))
    for h in HEADS:
        k[f"auroc_gen_head_{h}"] = stat(lambda i, h=h: 100 * T.auroc(head[h][i], good_a[i]), gp)
        k[f"ece_raw_gen_head_{h}"] = stat(lambda i, h=h: 100 * T.ece(head[h][i], good_a[i]), gp)
    k["auroc_gen_logprob_P(s_hat)"] = stat(lambda i: 100 * T.auroc(G["pshat"][P][i], good_a[i]), gp)
    k["auroc_gen_lenorm_conf"] = stat(lambda i: 100 * T.auroc(G["kap"][P][i], good_a[i]), gp)
    k["delta_auroc_glance_minus_gen_head_heldout_full"] = stat(
        lambda i: 100 * (T.auroc(kg[i], good_g[i]) - T.auroc(head["heldout_full"][i], good_a[i])), gp, seed=1)
    out["kappa_present_single_target"] = k
    return out, (A, Bq, ia, ib)


# ----------------------------------------------------------------------------------------------- two-knob items
def items(v1, gen, hs, Tpair):
    A, Bq, ia, ib = Tpair
    keep = np.array([(not r["present"]) or ("iou" in r) for r in A])
    I = {}
    kA = [A[k] for k in np.where(keep)[0]]; kB = [Bq[k] for k in np.where(keep)[0]]
    y = np.array([r["present"] for r in kA])
    G = gen_arrays(kB)
    I.update(t_y=y, t_img=np.array([img_int(r) for r in kA]),
             t_good_g=y & np.array([r.get("iou", 0) >= .5 for r in kA]), t_good_a=y & np.array([r.get("iou", 0) >= .5 for r in kB]),
             t_g=sig([r["p_logit"] for r in kA]), t_k=sig([r["iou_logit"] for r in kA]),
             t_ar_margin=G["margin"], t_ar_kap=G["kap"], t_ar_pshat=G["pshat"])
    for h in HEADS:
        I[f"t_head_{h}"] = hs[f"{h}__grefcoco_val_merged"][ib][keep]
    # C: gRefCOCO testA+testB single-target with box + no-target, images outside val
    val_imgs = {r["img"] for r in A}
    cv1 = [r for sp in ("grec_testA", "grec_testB") for r in rows(f"{v1}/{sp}.jsonl.gz")
           if r["img"] not in val_imgs and ((not r["present"]) or (r.get("n_targets") == 1 and r.get("gt_box") is not None))]
    cg = rows(f"{gen}/rows_arcal_gtest.jsonl.gz")
    ja, jb = join(cv1, cg)
    assert len(ja) == len(cv1) == len(cg), (len(ja), len(cv1), len(cg))
    CA = [cv1[i] for i in ja]; CB = [cg[i] for i in jb]
    yc = np.array([r["present"] for r in CA])
    GC = gen_arrays(CB)
    I.update(c_y=yc, c_img=np.array([img_int(r) for r in CA]),
             c_good_g=yc & np.array([r.get("iou", 0) >= .5 for r in CA]), c_good_a=yc & np.array([r.get("iou", 0) >= .5 for r in CB]),
             c_g=sig([r["p_logit"] for r in CA]), c_k=sig([r["iou_logit"] for r in CA]),
             c_ar_margin=GC["margin"], c_ar_kap=GC["kap"])
    for h in HEADS:
        I[f"c_head_{h}"] = hs[f"{h}__arcal_gtest"][jb]
    return I


def twoknob(I):
    y, img = I["t_y"], I["t_img"]
    cal, priors = T.calibrate(I)
    res = dict(n_test=int(len(y)), n_present=int(y.sum()), n_cal=int(len(I["c_y"])), n_cal_absent=int((~I["c_y"]).sum()),
               priors=priors, cal_box_error=dict(glance=float(1 - I["c_good_g"][I["c_y"]].mean()), gen=float(1 - I["c_good_a"][I["c_y"]].mean())),
               test_box_error=dict(glance=float(1 - I["t_good_g"][y].mean()), gen=float(1 - I["t_good_a"][y].mean())))
    tab = {}
    for pr, C in cal.items():
        tp = {}
        for cW in T.CW:
            row, acc = {}, {}
            for m, (eta, kap) in C.items():
                gd = I["t_good_g"] if m == "glance" else I["t_good_a"]
                cv = cost_vec(T.two_knob(eta, kap, cW), dict(y=y, good=gd), (1.0, 1.0, cW))
                acc[m] = cv
                row[f"{m}:two_knob"] = dict(cost=float(cv.mean()), ci95=boot_ci(cv, img))
            for m in C:
                if m != "glance":
                    d = acc["glance"] - acc[m]
                    row[f"delta_glance_minus_{m}"] = dict(mean=float(d.mean()), ci95=boot_ci(d, img, seed=2),
                                                          rel=float(d.mean() / acc[m].mean()))
            tp[f"cW={cW}"] = row
        tab[pr] = tp
    res["costs_cH=cR=1"] = tab
    # box-accuracy-equalised (ablation/twoknob_v3_boxeq.py): absent items + present items where both boxes agree in correctness
    S = (~y) | (I["t_good_g"] == I["t_good_a"])
    be = dict(n_subset=int(S.sum()), n_present_subset=int((S & y).sum()), n_dropped_present=int((y & ~S).sum()),
              box_error_subset=float(1 - I["t_good_g"][S & y].mean()))
    for pr, C in cal.items():
        tp = {}
        for cW in T.CW:
            acc = {}
            for m in ("glance", "ar", "ar+heldout_full"):
                eta, kap = C[m]
                gd = I["t_good_g"] if m == "glance" else I["t_good_a"]
                acc[m] = cost_vec(T.two_knob(eta, kap, cW), dict(y=y, good=gd), (1.0, 1.0, cW))[S]
            d = acc["glance"] - acc["ar+heldout_full"]
            tp[f"cW={cW}"] = {**{f"{m}:two_knob": float(v.mean()) for m, v in acc.items()},
                              "delta_glance_minus_ar+heldout_full": dict(mean=float(d.mean()), ci95=boot_ci(d, img[S], seed=2),
                                                                         rel=float(d.mean() / acc["ar+heldout_full"].mean()))}
        be[pr] = tp
    res["boxeq"] = be
    return res


# ----------------------------------------------------------------------------------------------- GUI
def gui(v1, gen):
    out = {}
    for sp in ("ss2", "venus"):
        if not os.path.exists(f"{gen}/rows_{sp}.jsonl.gz"):
            continue
        a, b = rows(f"{v1}/{sp}.jsonl.gz"), rows(f"{gen}/rows_{sp}.jsonl.gz")
        ia, ib = join(a, b)
        A = [a[i] for i in ia]; Bq = [b[i] for i in ib]
        y = np.array([r["present"] for r in A])
        g = np.array([zlib.crc32(r["img"].encode()) for r in A])
        ca = np.array([bool(r.get("center_in_box")) for r in A]); cb = np.array([bool(r.get("center_in_box")) for r in Bq])
        ref_a = sig([r["p_logit"] for r in A]) < .5
        ref_b = np.array([r["greedy_none"] for r in Bq])
        o = dict(n=len(A), n_v1=len(a), n_gen=len(b), n_present=int(y.sum()), n_absent=int((~y).sum()),
                 click_ungated=dict(glance=stat(lambda i: 100 * ca[i][y[i]].mean(), g), gen=stat(lambda i: 100 * cb[i][y[i]].mean(), g)),
                 click_gated=dict(glance=stat(lambda i: 100 * (ca & ~ref_a)[i][y[i]].mean(), g),
                                  gen_greedy=stat(lambda i: 100 * (cb & ~ref_b)[i][y[i]].mean(), g)),
                 false_refusal=dict(glance=float(ref_a[y].mean() * 100), gen_greedy=float(ref_b[y].mean() * 100)))
        if (~y).any():
            o["refusal_acc"] = dict(glance=stat(lambda i: 100 * ref_a[i][~y[i]].mean(), g),
                                    gen_greedy=stat(lambda i: 100 * ref_b[i][~y[i]].mean(), g))
            o["presence_auroc"] = dict(glance=100 * T.auroc(np.array([r["p_logit"] for r in A]), y),
                                       gen_pnone=100 * T.auroc(-np.array([r["p_none"] for r in Bq]), y))
        out[sp] = o
    return out


def venus_official(d):
    """official UI-Venus eval.py logs (<cfg>/<task>.json) -> per task acc and overall micro"""
    import glob
    r = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        o = json.load(open(f))["metrics"]["overall"]
        r[os.path.basename(f)[:-5]] = dict(correct=o["num_correct_action"], total=o["num_total"], wrong_format=o["wrong_format_num"],
                                           acc=100 * o["num_correct_action"] / o["num_total"])
    if r:
        c = sum(v["correct"] for v in r.values()); n = sum(v["total"] for v in r.values())
        r["overall_micro"] = 100 * c / n
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--venus", default="", help="dir with official-evaluator outputs: <dir>/{v1_raw_gated,gen_greedy,gen_pnone}")
    ap.add_argument("--out", default="outputs/gen_v1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hs = dict(np.load(f"{a.gen}/head_scores.npz", allow_pickle=True))
    gv = rows(f"{a.gen}/rows_grefcoco_val_merged.jsonl.gz")
    res = {}
    res["grefcoco_val_merged"], Tpair = grec(rows(f"{a.v1}/grec_val.jsonl.gz"), gv, hs)
    res["refcoco"] = refcoco(a.v1, a.gen, gv)
    I = items(a.v1, a.gen, hs, Tpair)
    np.savez_compressed(f"{a.out}/items.npz", **I)
    res["twoknob"] = twoknob(I)
    res["gui"] = gui(a.v1, a.gen)
    if a.venus:
        res["venus_official"] = {c: venus_official(f"{a.venus}/{c}") for c in sorted(os.listdir(a.venus))
                                 if os.path.isdir(f"{a.venus}/{c}")}
    for f in ("decode_check.json", "head_scores_info.json"):
        if os.path.exists(f"{a.gen}/{f}"):
            res[f.split(".")[0]] = json.load(open(f"{a.gen}/{f}"))
    if "head_scores_info" in res:
        for v in res["head_scores_info"].values():
            v.pop("curve", None)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "twoknob"}, indent=1)[:6000])
    for pr, tp in res["twoknob"]["costs_cH=cR=1"].items():
        for cw, row in tp.items():
            print(pr, cw, " ".join(f"{m.split(':')[0]}={v['cost']*100:.2f}" for m, v in row.items() if m.endswith("two_knob")),
                  "d(G-AR+head)=%.2f" % (100 * row["delta_glance_minus_ar+heldout_full"]["mean"]))


if __name__ == "__main__":
    main()
