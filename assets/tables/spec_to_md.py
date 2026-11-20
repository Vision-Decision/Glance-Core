"""Markdown version of a bench_table spec (same values, marks and winners)."""
import json, sys
from bench_table import fmt, winners
spec = json.load(open(sys.argv[1])); cols = spec["columns"]
out = ["| Category | Benchmark (metric) | " + " | ".join(f"**{cols[0]}**" if i == 0 else c for i, c in enumerate(cols)) + " |",
       "|---|---|" + "---|" * len(cols)]
prev = None
for r in spec["rows"]:
    f = r.get("fmt", "{:.1f}"); win = winners(r["vals"], f, r.get("better", "high"))
    marks = r.get("marks") or [""] * len(cols)
    cells = []
    for j, v in enumerate(r["vals"]):
        s = fmt(v, f) + (marks[j] if v is not None and marks[j] else "")
        cells.append(f"**{s}**" if j in win else s)
    cat = r["cat"].replace("\n", " ")
    note = r.get("note", "").replace("\n", " ")
    out.append(f"| {cat if cat != prev else ''} | {r['bench']} ({note}) | " + " | ".join(cells) + " |")
    prev = cat
print("\n".join(out))
