"""Paired v1.1 -> v1.2 row comparison for the four regressions (RefCOCO val, gRefCOCO N-acc, robust-ref zh, relational N-acc).
Rows are joined by order (same eval files, same deterministic batching) and checked on the expression text.

  python scripts/v12/regress_diag.py --v11 DIR --v12 DIR --out outputs/v1.2/results/regress_diag.json
Each set: net change, flips both ways, and the regressed rows broken down by simple text features (length, position /
relation words, numerals, category words, Chinese), so the cause can be tied to a data share of the v1.2 mixture.
"""
import argparse, collections, gzip, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ground_v2"))
from lexicon import EN, names_of, mentions  # noqa: E402

POS = re.compile(r"\b(left|right|top|bottom|middle|center|centre|front|back|behind|closest|nearest|far|furthest|farthest|"
                 r"first|second|third|upper|lower|corner|leftmost|rightmost|biggest|smallest|largest|tallest)\b|左|右|上|下|中间|前|后|最")
REL = re.compile(r"\b(next to|beside|near|on|under|above|below|holding|with|behind|in front of|between|left of|right of)\b|旁边|边上|上面|下面|拿着")
NUM = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|\d+)\b|[两三四五六七八九十]")
CATW = sorted({w.lower() for c in EN for w in names_of(c) if len(w) > 2 and not re.search(r"[一-鿿]", w)})


def load(p):
    return [json.loads(l) for l in gzip.open(p, "rt") if '"kind": "box"' in l]


def feats(t):
    f = []
    n = len(t.split()) if not re.search(r"[一-鿿]", t) else len(t) // 2
    f.append("len<=2" if n <= 2 else "len3-5" if n <= 5 else "len>=6")
    f.append("pos_word" if POS.search(t.lower()) else "no_pos_word")
    if REL.search(t.lower()):
        f.append("relation_word")
    if NUM.search(t.lower()):
        f.append("numeral")
    f.append("cat_word" if mentions(t, CATW) else "no_cat_word")
    return f


def correct(r):
    if r["present"]:
        return r["p_present"] >= 0.5 and r.get("iou", 0) >= 0.5
    return r["p_present"] < 0.5


def compare(a, b, name, subset=None):
    assert len(a) == len(b), (name, len(a), len(b))
    lost, gained = collections.Counter(), collections.Counter()
    n = nl = ng = 0
    lost_ex = []
    for x, y in zip(a, b):
        assert x["text"] == y["text"], (name, x["text"], y["text"])
        if subset and not subset(x):
            continue
        n += 1
        cx, cy = correct(x), correct(y)
        if cx and not cy:
            nl += 1
            for f in feats(x["text"]):
                lost[f] += 1
            if len(lost_ex) < 12:
                lost_ex.append(dict(text=x["text"], present=x["present"], p11=round(x["p_present"], 3), p12=round(y["p_present"], 3),
                                    iou11=round(x.get("iou", 0), 2), iou12=round(y.get("iou", 0), 2)))
        elif cy and not cx:
            ng += 1
            for f in feats(x["text"]):
                gained[f] += 1
    base = collections.Counter(f for x in a if (not subset or subset(x)) for f in feats(x["text"]))
    mode = collections.Counter()
    for x, y in zip(a, b):
        if subset and not subset(x):
            continue
        if correct(x) and not correct(y):
            if not x["present"]:
                mode["absent->accepted"] += 1
            elif y["p_present"] < 0.5:
                mode["present->rejected"] += 1
            else:
                mode["present->wrong_box"] += 1
    return dict(n=n, lost=nl, gained=ng, net_pts=round(100 * (ng - nl) / max(1, n), 2), lost_mode=dict(mode),
                lost_by_feature={k: dict(lost=v, gained=gained.get(k, 0), share_of_set=round(base[k] / n, 3),
                                         net_pts_within=round(100 * (gained.get(k, 0) - v) / max(1, base[k]), 2))
                                 for k, v in lost.most_common()},
                lost_examples=lost_ex)


ap = argparse.ArgumentParser()
ap.add_argument("--v11", required=True); ap.add_argument("--v12", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
res = {}
f = lambda d, n: load(os.path.join(d, n))
res["refcoco_val"] = compare(f(a.v11, "refcoco_val.jsonl.gz"), f(a.v12, "refcoco_val.jsonl.gz"), "refcoco_val")
g11, g12 = f(a.v11, "grefcoco_val_merged.jsonl.gz"), f(a.v12, "grefcoco_val_merged.jsonl.gz")
res["grefcoco_no_target"] = compare(g11, g12, "gref", subset=lambda r: not r["present"])
res["robust_zh"] = compare(f(a.v11, "robust_zh.jsonl.gz"), f(a.v12, "robust_zh.jsonl.gz"), "zh")
v11, v12 = f(a.v11, "robust_vague.jsonl.gz"), f(a.v12, "robust_vague.jsonl.gz")
res["relational_absent"] = compare(v11, v12, "rel", subset=lambda r: (r.get("meta") or {}).get("sub") == "relational" and not r["present"])
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
for k, v in res.items():
    print(k, {kk: v[kk] for kk in ("n", "lost", "gained", "net_pts", "lost_mode")})
    for fk, fv in list(v["lost_by_feature"].items())[:8]:
        print("   ", fk, fv)
