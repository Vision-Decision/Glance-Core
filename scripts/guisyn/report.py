"""Final batch report (prints JSON): per-app yield, throughput, disk use, 1M-element projection.
  python report.py   (on the host)"""
import glob, json, os, subprocess, time
from collections import Counter, defaultdict

st = json.load(open("/data/guisyn/stage/stats.json"))
jobs = sorted(glob.glob("/data/guisyn/raw/jobs/*.jsonl")) + sorted(glob.glob("/data/guisyn/rawweb/jobs/*.jsonl"))
ts, el = defaultdict(list), Counter()
for f in jobs:
    src = "web" if "rawweb" in f else "desktop"
    for l in open(f):
        r = json.loads(l)
        ts[src].append(r["t"])
        el[src] += sum(1 for e in r["elements"] if not e.get("uncaptioned"))
thr = {}
for s, t in ts.items():
    h = (max(t) - min(t)) / 3600
    thr[s] = dict(screenshots=len(t), hours=round(h, 2), shots_per_h=round(len(t) / h), elements=el[s],
                  elements_per_h=round(el[s] / h), elements_per_shot=round(el[s] / len(t), 1))
du = lambda p: subprocess.run(["du", "-sb", p], capture_output=True, text=True).stdout.split()[0]
disk = {p: round(int(du(f"/data/guisyn/{p}")) / 1e9, 2) for p in ["raw", "rawweb", "stage"]}
q = st["questions"]; n = st["screenshots"]
bytes_stage = sum(st["bytes_per_source"].values())
out = dict(stats=dict(screenshots=n, questions=q, kinds=st["kinds"], negative_frac=round(st["negative_frac"], 3)),
           throughput=thr, disk_gb=disk,
           per_app={a: dict(shots=v["screenshots"], q=v["questions"], neg=v.get("q_negative_hard", 0) + v.get("q_negative_easy", 0),
                            unc=v.get("uncaptioned_icons", 0)) for a, v in st["per_app"].items()},
           projection_1M_elements=dict(
               desktop_hours_at_current_rate=round(1e6 / thr["desktop"]["elements_per_h"], 1),
               web_hours_at_current_rate=round(1e6 / thr["web"]["elements_per_h"], 1) if "web" in thr else None,
               stage_gb_per_1M_questions=round(bytes_stage / q * 1e6 / 1e9, 1)),
           resolutions=st["resolutions"])
print(json.dumps(out, indent=1))
