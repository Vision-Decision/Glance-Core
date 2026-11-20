"""Score description-shift rows (eval.py --rows_dir output of Glance checkpoints, or base_gen.py rows).

Per slice (shift x kind, and language):
  acc50          present: predicted present (P>=0.5) and IoU>=0.5            (gated Acc@0.5)
  acc50_box      present: IoU>=0.5 regardless of presence                     (ungated)
  false_absent   present: P(present)<0.5;  false_absent_conf: P(present)<0.2  (confident "absent")
  N_acc          absent : P(present)<0.5
  overconf_err   all    : the answer (box if P>=0.5, else "absent") is wrong while its confidence > 0.8
                          (box confidence = P(present)*P(IoU>=.5), absent confidence = 1-P(present))
  ece            all    : ECE (15 bins) of box confidence P(present)*P(IoU>=.5) vs "present and IoU>=.5"
  ece_presence   all    : ECE of P(present) vs present
  all_correct    per group (gid): every description of the target answered correctly (present: gated hit; absent: P<0.5)
Generative rows (base_gen.py) use pred_present / p_present_gen; their box confidence is P_gen(present) (no IoU head).
Usage: python score.py --rows name=dir ... --set rg --out scores.json [--T name=presence_T]
"""
import argparse, collections, gzip, json, math
import numpy as np


def ece(p, y, n_bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(n_bins) if (b == k).any()))


def load(path, T=None):
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    for r in rows:
        if "out" in r:  # generative
            r["p"] = r.get("p_present_gen", float(r["pred_present"]))
            r["pred"] = bool(r["pred_present"])
            r["conf_box"] = r["p"]
        else:
            p = r["p_present"]
            if T and "p_logit" in r:
                p = 1 / (1 + math.exp(-r["p_logit"] / T))
            r["p"] = p; r["pred"] = p >= 0.5; r["conf_box"] = p * r.get("p_iou50", 1.0)
        r["hit"] = r.get("iou", 0.0) >= 0.5
        r["correct"] = (r["pred"] and r["hit"]) if r["present"] else (not r["pred"])
        r["conf_ans"] = r["conf_box"] if r["pred"] else 1 - r["p"]
    return rows


def metrics(rows):
    pres = [r for r in rows if r["present"]]; ab = [r for r in rows if not r["present"]]
    m = dict(n=len(rows), n_present=len(pres), n_absent=len(ab))
    mean = lambda xs: float(np.mean(xs)) if len(xs) else None
    if pres:
        m["acc50"] = mean([r["pred"] and r["hit"] for r in pres])
        m["acc50_box"] = mean([r["hit"] for r in pres])
        m["false_absent"] = mean([not r["pred"] for r in pres])
        m["false_absent_conf"] = mean([r["p"] < 0.2 for r in pres])
    if ab:
        m["N_acc"] = mean([not r["pred"] for r in ab])
    m["overconf_err"] = mean([(not r["correct"]) and r["conf_ans"] > 0.8 for r in rows])
    m["ece"] = ece([r["conf_box"] for r in rows], [r["present"] and r["hit"] for r in rows])
    m["ece_presence"] = ece([r["p"] for r in rows], [r["present"] for r in rows])
    m["acc"] = mean([r["correct"] for r in rows])
    return m


def group_all_correct(rows, key="gid"):
    g = collections.defaultdict(list)
    for r in rows:
        g[r["meta"][key]].append(r["correct"])
    return float(np.mean([all(v) for v in g.values()])) if g else None, len(g)


def score(rows, slice_keys):
    out = dict(all=metrics(rows))
    for key in slice_keys:
        g = collections.defaultdict(list)
        for r in rows:
            g["/".join(str(r["meta"].get(k)) for k in key.split("+"))].append(r)
        out[key] = {k: metrics(v) for k, v in sorted(g.items())}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="name=rows_dir")
    ap.add_argument("--set", default="rg")
    ap.add_argument("--T", nargs="*", default=[], help="name=presence temperature")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Ts = dict(x.split("=") for x in a.T)
    res = {}
    for spec in a.rows:
        name, d = spec.split("=", 1)
        rows = load(f"{d}/{a.set}.jsonl.gz", float(Ts[name]) if name in Ts else None)
        if a.set == "rg":
            s = score(rows, ["kind+shift", "shift", "lang", "kind+shift+lang"])
            pres = [r for r in rows if r["meta"]["kind"] == "present"]
            s["all_correct_present6"], s["n_groups_present"] = group_all_correct(pres)
            s["all_correct_absent3"], _ = group_all_correct([r for r in rows if r["meta"]["kind"] == "absent"])
            s["all_correct_unique4"], _ = group_all_correct([r for r in rows if r["meta"]["kind"] == "unique"])
            shifted = [r for r in rows if r["meta"]["shift"] != "orig"]
            s["shifted"] = metrics(shifted)
        else:  # glance-robust-ref (other agent's set): meta = {id, type, sub, lang}
            s = score(rows, ["sub", "lang", "sub+lang"])
        res[name] = s
    json.dump(res, open(a.out, "w"), indent=1)
    f = lambda v: "   -  " if v is None else f"{100 * v:6.1f}"
    print(f"{'model':14s} {'slice':22s} {'Acc.5':>6s} {'FAbs':>6s} {'FAbs.2':>6s} {'N-acc':>6s} {'OvErr':>6s} {'ECE':>6s}")
    for name, s in res.items():
        sl = s.get("kind+shift", s.get("sub"))
        for k, m in [("ALL", s["all"])] + list(sl.items()):
            print(f"{name:14s} {k:22s} {f(m.get('acc50'))} {f(m.get('false_absent'))} {f(m.get('false_absent_conf'))} "
                  f"{f(m.get('N_acc'))} {f(m.get('overconf_err'))} {f(m.get('ece'))}")
        if "all_correct_present6" in s:
            print(f"{name:14s} all-correct present(6 desc) {f(s['all_correct_present6'])}  absent(3) {f(s['all_correct_absent3'])}"
                  f"  unique(4) {f(s['all_correct_unique4'])}")
