"""Exp 5: question-only shortcut probe for the presence gate (CPU, minutes).

A text-only classifier sees ONLY the referring expression (no image) and predicts P(present), trained on the box
questions of a Glance training mixture. Whatever it gets right on an eval set is answerable without looking: the size of
the text shortcut. We then ask whether Glance's known presence failures line up with this text prior:
  * attribute swaps (count "the two trains", size, colour, action, position) on single-instance images (B set)
  * single-instance superlatives / positions ("the leftmost cat", "the cat on the left") (A set)
  * gRefCOCO val merged, robust-ref (zh / paraphrase / function / vague), Glance-Data diagnostic groups.

  python arch/shortcut/probe.py --train v1.2/train.jsonl[,train_base.jsonl] --repo . --diag diagnostic.jsonl \
      --diag_rows v1.1_rows.jsonl.gz --out outputs/arch/shortcut
"""
import argparse, collections, gzip, json, os, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import roc_auc_score


def box_qs(path, cap=0, seed=0):
    X, y, neg = [], [], []
    op = gzip.open if path.endswith(".gz") else open
    for l in op(path, "rt"):
        r = json.loads(l)
        if not r.get("img"):
            continue
        for q in r.get("questions", []):
            if q.get("type", "box") != "box" or not q.get("text"):
                continue
            X.append(q["text"]); y.append(int(bool(q["present"]))); neg.append(q.get("neg") or "")
    if cap and len(X) > cap:
        idx = random.Random(seed).sample(range(len(X)), cap)
        X = [X[i] for i in idx]; y = [y[i] for i in idx]; neg = [neg[i] for i in idx]
    return X, np.array(y), neg


def boot_auc(y, s, n=1000, seed=0, groups=None):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y))
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(ug))]
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([members[p] for p in pick])
        if len(set(y[idx])) == 2:
            bs.append(roc_auc_score(y[idx], s[idx]))
    return float(roc_auc_score(y, s)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def mean_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    if not len(x):
        return None, [None, None]
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="comma list of training mixtures (jsonl records); one probe each")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--diag", default="")
    ap.add_argument("--diag_rows", default="", help="Glance rows on the diagnostic set (same order) for alignment")
    ap.add_argument("--cap", type=int, default=1500000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = a.repo
    # ------------------------------------------------------------- eval sets
    ev = {}
    for name in ["grefcoco_val_merged", "robust_zh", "robust_para", "robust_func", "robust_vague"]:
        p = f"{R}/data/prep/eval_{name}.jsonl"
        if os.path.exists(p):
            X, y, recs = [], [], []
            for l in open(p):
                r = json.loads(l)
                for q in r["questions"]:
                    X.append(q["text"]); y.append(int(bool(q["present"]))); recs.append(r.get("image_id", r.get("file")))
            ev[name] = (X, np.array(y), recs)
    B = json.load(open(f"{R}/outputs/probes/ambiguity_attr/B_attribute_glance_v1.json"))["items"]
    A = json.load(open(f"{R}/outputs/probes/ambiguity_attr/A_ambiguity_glance_v1.json"))["items"]
    diag = [json.loads(l) for l in open(a.diag)] if a.diag else []
    drows = [json.loads(l) for l in gzip.open(a.diag_rows, "rt")] if a.diag_rows else []
    res = dict(sources=dict(train=a.train, diag=a.diag, diag_rows=a.diag_rows,
                            B="outputs/probes/ambiguity_attr/B_attribute_glance_v1.json",
                            A="outputs/probes/ambiguity_attr/A_ambiguity_glance_v1.json"), probes={})
    for tp in a.train.split(","):
        X, y, neg = box_qs(tp, a.cap)
        tag = os.path.basename(tp).replace(".jsonl", "")
        print(f"[probe] {tag}: {len(X)} box questions, present rate {y.mean():.3f}", flush=True)
        clf = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=400000),
                                       TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, sublinear_tf=True, max_features=400000)),
                            LogisticRegression(C=2.0, max_iter=2000))
        # held-out in-distribution check (10%)
        idx = np.random.default_rng(0).permutation(len(X)); cut = len(X) // 10
        te, tr = idx[:cut], idx[cut:]
        clf.fit([X[i] for i in tr], y[tr])
        P = dict(n_train=len(tr), present_rate=float(y.mean()))
        s = clf.predict_proba([X[i] for i in te])[:, 1]
        P["indist_auroc"], P["indist_auroc_ci95"] = boot_auc(y[te], s)
        by = collections.defaultdict(list)
        for i, v in zip(te, s):
            if not y[i]:
                by[neg[i] or "none"].append(v)
        P["indist_neg_kinds_mean_p"] = {k: (float(np.mean(v)), len(v)) for k, v in by.items() if len(v) >= 30}
        # eval sets
        P["eval"] = {}
        for name, (Xe, ye, ge) in ev.items():
            se = clf.predict_proba(Xe)[:, 1]
            auc, ci = boot_auc(ye, se, groups=np.array([str(g) for g in ge]))
            P["eval"][name] = dict(n=len(ye), auroc=auc, auroc_ci95=ci, mean_p_present=float(se[ye == 1].mean()) if (ye == 1).any() else None,
                                   mean_p_absent=float(se[ye == 0].mean()) if (ye == 0).any() else None)
        # B: attribute swaps. per type: text-only AUROC (correct vs swapped), mean text-P on swaps, and whether Glance's
        # false accepts are the swaps the text prior likes (AUROC of text-P for Glance accept among swaps)
        Bq = [(it["image_id"], q) for it in B for q in it["questions"]]
        sB = clf.predict_proba([q["text"] for _, q in Bq])[:, 1]
        P["B_attr"] = {}
        for t in sorted({q["type"] for _, q in Bq} - {"bare"}):
            m = np.array([q["type"] == t for _, q in Bq])
            yy = np.array([int(q["present"]) for _, q in Bq])[m]; ss = sB[m]
            gp = np.array([q.get("p", np.nan) for _, q in Bq])[m]
            sw = yy == 0
            d = dict(n=int(m.sum()), n_swap=int(sw.sum()))
            d["text_auroc_correct_vs_swap"], d["text_auroc_ci95"] = boot_auc(yy, ss, groups=np.array([g for (g, _), mm in zip(Bq, m) if mm]))
            d["text_p_on_swaps"] = mean_ci(ss[sw]); d["text_p_on_correct"] = mean_ci(ss[~sw])
            d["glance_FA_0.5"] = mean_ci((gp[sw] >= 0.5).astype(float))
            acc = (gp[sw] >= 0.5).astype(int)
            d["text_p_predicts_glance_FA_auroc"] = boot_auc(acc, ss[sw])[0] if 0 < acc.sum() < len(acc) else None
            d["spearman_text_vs_glance_logit_on_swaps"] = float(np.corrcoef(np.argsort(np.argsort(ss[sw])),
                                                             np.argsort(np.argsort(gp[sw])))[0, 1]) if sw.sum() > 3 else None
            P["B_attr"][t] = d
        # A: single-instance superlative / position phrasings (all present) vs multi-instance
        Aq = [(it["kind"], q) for it in A for q in it["questions"] if "p" in q]
        sA = clf.predict_proba([q["text"] for _, q in Aq])[:, 1]
        P["A_ambig"] = {}
        for key in sorted({f"{k}/{q['tag']}" for k, q in Aq}):
            m = np.array([f"{k}/{q['tag']}" == key for k, q in Aq])
            gp = np.array([q["p"] for _, q in Aq])[m]
            P["A_ambig"][key] = dict(n=int(m.sum()), text_p=mean_ci(sA[m]))
            P["A_ambig"][key]["glance_reject_0.5"] = mean_ci((gp < 0.5).astype(float))
            P["A_ambig"][key]["spearman_text_vs_glance"] = float(np.corrcoef(np.argsort(np.argsort(sA[m])), np.argsort(np.argsort(gp)))[0, 1]) if m.sum() > 3 else None
        # diagnostic set groups (Glance-Data): by qtype x present, with Glance v1.1 acceptance if rows given
        if diag:
            dq = [(r["image_id"], q) for r in diag for q in r["questions"]]
            sD = clf.predict_proba([q["text"] for _, q in dq])[:, 1]
            al = len(drows) == len(dq) and all(d["text"] == q["text"] for d, (_, q) in zip(drows, dq))
            P["diag_aligned_with_rows"] = al
            G = collections.defaultdict(list)
            for i, (_, q) in enumerate(dq):
                k = f"{q.get('qtype')}/{'pos' if q['present'] else 'neg:' + str(q.get('neg'))}"
                G[k].append(i)
            P["diag"] = {}
            for k, ii in sorted(G.items()):
                if len(ii) < 30:
                    continue
                ii = np.array(ii)
                d = dict(n=len(ii), text_p=float(sD[ii].mean()))
                if al:
                    gp = np.array([drows[i]["p_present"] for i in ii])
                    d["glance_accept_0.5"] = float((gp >= 0.5).mean())
                    d["spearman_text_vs_glance"] = float(np.corrcoef(np.argsort(np.argsort(sD[ii])), np.argsort(np.argsort(gp)))[0, 1])
                P["diag"][k] = d
            yD = np.array([int(q["present"]) for _, q in dq])
            P["diag_overall_text_auroc"], P["diag_overall_text_auroc_ci95"] = boot_auc(yD, sD, groups=np.array([g for g, _ in dq]))
        res["probes"][tag] = P
        print(json.dumps({k: v for k, v in P.items() if k in ("indist_auroc", "eval")}, indent=0)[:2500], flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
