"""Per-question-type set F1 on the glance-ground-v2 diagnostic split from row dumps (scripts/multibox/vm_diag.sh).

python scripts/multibox/diag_table.py base=rows_base/diag.jsonl.gz jointnb=rows_jointnb/diag.jsonl.gz [--thr 0.2]
Columns: each run decoded with the pre-registered rule (P(match) >= 0.5) and, for multi-box runs, the count-guided rule
(threshold for the >=4 bucket from calib_mb). Rows: qtype (Glance-Data kind) x lang, plus summary lines.
"""
import argparse, collections, gzip, json, os
import numpy as np

code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval.py")).read()
ns = dict(np=np, collections=collections)
exec(code[code.index("def ece("):code.index("@torch.no_grad()")], ns)
exec(code[code.index("def _iou("):code.index("def summarize_grec")], ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--thr", type=float, default=0.2)
    a = ap.parse_args()
    cols = {}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        rows = [json.loads(l) for l in gzip.open(path, "rt")]
        multi = any(r.get("cands") for r in rows)
        g = collections.defaultdict(list)
        for r in rows:
            m = r.get("meta") or {}
            g[f"{m.get('qtype')}/{m.get('lang') or 'en'}"].append(r)
        for mode, thr in ([("thr", 0.5), ("count", a.thr)] if multi else [("thr", 0.5)]):
            out = {}
            for k, rr in sorted(g.items()):
                out[k] = ns["summarize_multi"](rr, thr=thr, mode=mode)["all"]["f1"]
            sm = ns["summarize_multi"](rows, thr=thr, mode=mode)
            out.update({"ALL (set F1)": sm["all"]["f1"], "multi F1": sm["multi"]["f1"],
                        "multi count acc": sm["multi"]["count_acc"], "single F1": sm["single"]["f1"],
                        "no-target N-acc": sm.get("N_acc")})
            cols[f"{name} ({mode})"] = out
    keys = list(next(iter(cols.values())))
    print("| type | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for k in keys:
        print(f"| {k} | " + " | ".join("-" if cols[c].get(k) is None else f"{100 * cols[c][k]:.1f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
