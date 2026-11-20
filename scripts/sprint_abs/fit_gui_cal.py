"""Fit the GUI presence recalibrators used by the official VenusBench wrapper on guisyn-fit rows only.
python scripts/sprint_abs/fit_gui_cal.py guisyn.jsonl.gz platt_out.json stack_out.json"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_rows, guisyn_split, fit_presence  # noqa: E402

rows, po, so = sys.argv[1:4]
fit = [r for r in load_rows(rows) if guisyn_split(r) == "fit"]
pc, pi = fit_presence(fit)
a, b = pc["platt"].m.b
json.dump(dict(mu=[0] * 5, sd=[1] * 5, b=[float(a), 0, 0, 0, 0, float(b)], fit="Platt, guisyn fit apps", n=len(fit)), open(po, "w"))
m = pc["multi"]
json.dump(dict(mu=list(map(float, m.mu)), sd=list(map(float, m.sd)), b=list(map(float, m.m.b)), fit="5-feature, guisyn fit apps",
               n=len(fit)), open(so, "w"))
print("platt", a, b, "n", len(fit))
