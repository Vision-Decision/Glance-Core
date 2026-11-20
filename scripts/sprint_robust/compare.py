"""Aggregate sprint runs into markdown tables + summary.json.
Inputs: outputs/sprint_robust/runs/<name>/{eval_results.json, rows/rg.jsonl.gz}; the diag base rows.
Usage: python compare.py --runs e_v1 c0_ctrl r1_zhpara ... --out outputs/sprint_robust/summary
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import load, metrics, group_all_correct  # noqa: E402

R = "outputs/sprint_robust/runs"
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--base", default=f"{R}/diag/base")
ap.add_argument("--out", default="outputs/sprint_robust/summary")
ap.add_argument("--exclude_texts", default="", help="json list of training texts: score only eval queries never seen verbatim")
ap.add_argument("--calibrated", action="store_true", help="apply each run's fitted presence temperature to rg rows")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

SL = [("orig", lambda m: m["kind"] == "present" and m["shift"] == "orig"),
      ("zh", lambda m: m["kind"] == "present" and m["shift"] == "zh"),
      ("para", lambda m: m["kind"] == "present" and m["shift"] == "para"),
      ("func+attr", lambda m: m["kind"] == "present" and m["shift"] == "func"),
      ("vague", lambda m: m["kind"] == "present" and m["shift"] in ("vague", "zh_vague")),
      ("func-only", lambda m: m["kind"] == "unique" and m["shift"] == "func_only"),
      ("generic", lambda m: m["kind"] == "unique" and m["shift"] == "generic"),
      ("absent(g)", lambda m: m["kind"] == "absent"),
      ("absent-func", lambda m: m["kind"] == "absent_func")]


EXCL = set(json.load(open(a.exclude_texts))) if a.exclude_texts else set()


def rg_summary(rows):
    if EXCL:
        rows = [r for r in rows if r["text"].strip().lower() not in EXCL]
    s = {}
    for name, f in SL:
        s[name] = metrics([r for r in rows if f(r["meta"])])
    shifted = [r for r in rows if r["meta"]["shift"] != "orig"]
    s["shifted_all"] = metrics(shifted)
    s["shifted_present"] = metrics([r for r in shifted if r["present"]])
    s["all"] = metrics(rows)
    s["allc_present6"], _ = group_all_correct([r for r in rows if r["meta"]["kind"] == "present"])
    s["allc_unique4"], _ = group_all_correct([r for r in rows if r["meta"]["kind"] == "unique"])
    s["allc_absent3"], _ = group_all_correct([r for r in rows if r["meta"]["kind"] == "absent"])
    return s


def std_summary(ev):
    s, S = {}, ev.get("sets", {})
    for k in ("refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocog_val"):
        if k in S:
            s[k] = S[k].get("acc50")
    g = S.get("grefcoco_val_merged")
    if g:
        for k in ("N_acc", "T_acc_single", "auroc_presence", "ece_presence"):
            s["gref_" + k] = g.get(k)
    od = S.get("opendecisions")
    if od:
        c = od.get("cal") or od.get("raw") or {}
        s["od_capability"], s["od_calibration"] = c.get("capability"), c.get("calibration")
    for k in ("ss2", "sspro", "venus"):
        if k in S:
            raw = S[k].get("raw", S[k])
            s[k + "_click"] = raw.get("click_acc")
            if k == "venus":
                s["venus_refusal"] = raw.get("refusal_acc")
                fg = {g: v["click_acc"] for g, v in raw.get("by_group", {}).items() if "func" in g.lower()}
                if fg:
                    s["venus_functional"] = float(np.mean(list(fg.values())))
    return s


res = {}
if os.path.exists(f"{a.base}/rg.jsonl.gz"):
    res["base_gen"] = dict(rg=rg_summary(load(f"{a.base}/rg.jsonl.gz")), std={})
for n in a.runs:
    d = f"{R}/{n}"
    ev = json.load(open(f"{d}/eval_results.json")) if os.path.exists(f"{d}/eval_results.json") else {}
    T = ev.get("temps", {}).get("presence") if a.calibrated else None
    rp = f"{d}/rows/rg.jsonl.gz"
    res[n] = dict(rg=rg_summary(load(rp, T)) if os.path.exists(rp) else {}, std=std_summary(ev), T=T)
# ---- glance-robust-ref (independent eval set built by another agent): per set, per sub-type
EXTRA = {"e_v1": [f"{R}/diag/v1"], "base_gen": [f"{R}/diag/base"]}


def find_rows(n, s):
    for d in [f"{R}/{n}/rows", f"{R}/{n}_rob/rows"] + EXTRA.get(n, []):
        if os.path.exists(f"{d}/{s}.jsonl.gz"):
            return f"{d}/{s}.jsonl.gz"


ROB = ["robust_zh", "robust_para", "robust_func", "robust_vague"]
for n in res:
    rr = {}
    for s in ROB:
        p = find_rows(n, s)
        if p:
            rows = load(p, res[n].get("T"))
            if EXCL:
                rows = [r for r in rows if r["text"].strip().lower() not in EXCL]
            subs = {}
            for r in rows:
                subs.setdefault(r["meta"]["sub"], []).append(r)
            rr[s] = dict(all=metrics(rows), sub={k: metrics(v) for k, v in sorted(subs.items())})
    res[n]["robust_ref"] = rr
json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1)

f = lambda v, d=1: "–" if v is None else f"{100 * v:.{d}f}"
L = ["## Description-shift eval (robust-groups, VAL) — Acc@0.5 (gated) / false-absence", "",
     "| model | " + " | ".join(n for n, _ in SL[:7]) + " | shifted Acc | shifted FA | conf-FA(<.2) | N-acc(abs) | N-acc(abs-func) | overconf err | ECE | all-correct(6) | all-correct(unique4) |",
     "|" + "---|" * 17]
for n, r in res.items():
    g = r["rg"]
    if not g:
        continue
    cells = [f"{f(g[s]['acc50'])} / {f(g[s]['false_absent'])}" for s, _ in SL[:7]]
    L.append(f"| {n} | " + " | ".join(cells) + f" | {f(g['shifted_present']['acc50'])} | {f(g['shifted_present']['false_absent'])} | "
             f"{f(g['shifted_present']['false_absent_conf'])} | {f(g['absent(g)']['N_acc'])} | {f(g['absent-func']['N_acc'])} | "
             f"{f(g['all']['overconf_err'])} | {f(g['all']['ece'])} | {f(g['allc_present6'])} | {f(g['allc_unique4'])} |")
L += ["", "## Per-shift ECE (box confidence) / overconfident-error rate", "",
      "| model | " + " | ".join(n for n, _ in SL) + " |", "|" + "---|" * (len(SL) + 1)]
for n, r in res.items():
    g = r["rg"]
    if g:
        L.append(f"| {n} | " + " | ".join(f"{f(g[s]['ece'])} / {f(g[s]['overconf_err'])}" for s, _ in SL) + " |")
keys = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocog_val", "gref_N_acc", "gref_T_acc_single",
        "gref_auroc_presence", "gref_ece_presence", "od_capability", "od_calibration", "ss2_click", "sspro_click", "venus_click",
        "venus_refusal", "venus_functional"]
L += ["", "## Standard benchmarks", "", "| model | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
for n, r in res.items():
    s = r["std"]
    if s:
        L.append(f"| {n} | " + " | ".join(("–" if s.get(k) is None else (f"{s[k]:.1f}" if k.startswith("od_") else f"{100 * s[k]:.1f}"))
                                         for k in keys) + " |")
L += ["", "## glance-robust-ref (independent set) — Acc@0.5 / false-absence (present), N-acc (absent), overconf err", "",
      "| model | zh | para | func | vague | func: function_only | func: function_attr | vague: generic | vague: relational | vague: generic_attr | N-acc zh/para/func/vague | overconf err (all 4) |",
      "|" + "---|" * 12]
for n, r in res.items():
    rr = r.get("robust_ref") or {}
    if len(rr) < 4:
        continue
    c = lambda m: f"{f(m.get('acc50'))} / {f(m.get('false_absent'))}"
    sub = lambda s, k: c(rr[s]["sub"].get(k, {}))
    oc = np.mean([rr[s]["all"]["overconf_err"] for s in ROB])
    L.append(f"| {n} | " + " | ".join(c(rr[s]["all"]) for s in ROB) + f" | {sub('robust_func', 'function_only')} | {sub('robust_func', 'function_attr')} | "
             f"{sub('robust_vague', 'generic')} | {sub('robust_vague', 'relational')} | {sub('robust_vague', 'generic_attr')} | "
             + "/".join(f(rr[s]["all"].get("N_acc"), 0) for s in ROB) + f" | {f(oc)} |")
open(f"{a.out}/tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
