"""PREREG pass/fail table for v1.2 (outputs/v1.2/PREREG.md) from the post-training eval files.

  python scripts/v12/report.py --post <dir with diag.json robust_scores.json eval_results.json eval_results_gui.json>
                               [--s2 <dir with results.json from scripts/sprint_abs/analyze.py>] --out outputs/v1.2/results.json
Group D numbers come from the S2 pipeline output when given (keys printed if not found, never guessed).
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--post", required=True); ap.add_argument("--s2", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
L = lambda f: json.load(open(os.path.join(a.post, f))) if os.path.exists(os.path.join(a.post, f)) else None
diag, rob, ev, gui = L("diag.json"), L("robust_scores.json"), L("eval_results.json"), L("eval_results_gui.json")
rows = []


def add(cid, name, val, op, thr, ref):
    ok = None if val is None else (val <= thr if op == "<=" else val >= thr)
    rows.append(dict(id=cid, name=name, value=val, op=op, threshold=thr, v11=ref, passed=ok))


if rob:
    sub = rob["v1.2"]["sets"]
    g = lambda s, k, m: 100 * sub[s]["sub"][k][m] if k in sub[s]["sub"] and m in sub[s]["sub"][k] else None
    add("A1", "robust-ref function_only false-absent %", g("func", "function_only", "false_absent"), "<=", 10, 36.8)
    add("A2", "robust-ref generic false-absent %", g("vague", "generic", "false_absent"), "<=", 10, 44.1)
    add("A3", "robust-ref relational false-absent %", g("vague", "relational", "false_absent"), "<=", 10, 50.7)
    add("A4", "robust-ref relational gated Acc@0.5", g("vague", "relational", "acc50"), ">=", 40, 13.6)
if diag:
    d = diag["per_kind"]
    add("B1", "diag neg_count N-acc", d["neg_count/all"]["n_acc"], ">=", 65, 46.9)
    add("B2", "diag pos_pos_sole accepted %", d["pos_pos_sole/all"]["recall"], ">=", 70, 33.8)
    add("B3", "diag cat_only gated Acc@0.5", d["cat_only/all"]["acc50"], ">=", 60, 44.1)
    add("B4", "diag hneg_attribute N-acc", d["hneg_attribute/all"]["n_acc"], ">=", 68.4, 68.4)
    add("B5", "diag hneg_relation N-acc", d["hneg_relation/all"]["n_acc"], ">=", 30.0, 30.0)
if ev:
    s = ev["sets"]
    splits = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocoplus_testA", "refcocoplus_testB",
              "refcocog_val", "refcocog_test"]
    add("C1", "RefCOCO val Acc@0.5", 100 * s["refcoco_val"]["acc50"], ">=", 94, 94.9)
    add("C2", "RefCOCO 8-split avg Acc@0.5", 100 * sum(s[k]["acc50"] for k in splits) / 8, ">=", 92.5, 92.8)
    gm = s["grefcoco_val_merged"]
    add("C3", "gRefCOCO val N-acc (thr 0.5)", 100 * gm["N_acc"], ">=", 86, 86.0)
    add("C4", "gRefCOCO val presence ECE (calibrated)", (gm.get("calibrated") or gm)["ece_presence"], "<=", 0.03, 0.036)
    od = s["opendecisions"]
    add("C8", "OpenDecisions capability (raw)", od["raw"]["capability"], ">=", 66, 67.8)
if gui or ev:
    s = dict((ev or {}).get("sets", {}), **(gui or {}).get("sets", {}))
    g = lambda k, m: 100 * s[k]["raw"][m] if k in s else None
    add("C5", "VenusBench refusal_acc (summarize_gui)", g("venus", "refusal_acc"), ">=", 81, 82.2)
    add("C6", "VenusBench grounding click_acc (summarize_gui)", g("venus", "click_acc"), ">=", 53, 55.2)
    add("C7", "ScreenSpot-v2 click_acc", g("ss2", "click_acc"), ">=", 88, 89.2)
if a.s2 and os.path.exists(f"{a.s2}/results.json"):
    r = json.load(open(f"{a.s2}/results.json"))
    print("S2 result keys:", list(r)[:40])
allsets = dict((ev or {}).get("sets", {}), **(gui or {}).get("sets", {}))
sp = allsets.get("sspro", {})
out = dict(rows=rows, extra=dict(sspro=sp.get("raw", {}).get("click_acc"), sspro_zoom=sp.get("zoom", {}).get("click_acc")))
json.dump(out, open(a.out, "w"), indent=1)
f = lambda v: "-" if v is None else (f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}")
for x in rows:
    print(f"{x['id']:3s} {x['name']:45s} {f(x['value']):>7s} {x['op']} {x['threshold']:<6} (v1.1 {x['v11']})  "
          f"{'PASS' if x['passed'] else 'FAIL' if x['passed'] is not None else 'n/a'}")
