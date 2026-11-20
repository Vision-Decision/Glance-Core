"""Aggregate stats over built shards -> {stage}/stats.json (uploaded with the data) + throughput from raw jobs."""
import glob, json, os, sys, time
from collections import Counter, defaultdict

stage = sys.argv[1] if len(sys.argv) > 1 else "/data/guisyn/stage"
per_app = defaultdict(Counter)
res = Counter()
kinds = Counter()
for f in glob.glob(f"{stage}/shards/*/*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        k = r["app"] if r["source"] == "desktop" else "web"
        c = per_app[k]
        c["screenshots"] += 1
        c["uncaptioned_icons"] += len(r.get("uncaptioned", []))
        for q in r["questions"]:
            c["q_" + q["kind"]] += 1
            c["questions"] += 1
            kinds[q["kind"]] += 1
        res[f"{r['source']}:{r['W']}x{r['H']}@{r['meta'].get('scale', 1)}"] += 1
sizes = {os.path.basename(d): sum(os.path.getsize(x) for x in glob.glob(f"{d}/*")) for d in glob.glob(f"{stage}/shards/*")}
tot = sum(c["screenshots"] for c in per_app.values())
out = dict(updated=time.strftime("%F %T UTC", time.gmtime()), screenshots=tot,
           questions=sum(kinds.values()), kinds=kinds, negative_frac=(kinds["negative_hard"] + kinds["negative_easy"]) / max(1, sum(kinds.values())),
           per_app={k: dict(v) for k, v in sorted(per_app.items())}, resolutions=dict(res.most_common()),
           bytes_per_source=sizes)
json.dump(out, open(f"{stage}/stats.json.tmp", "w"), indent=1)
os.rename(f"{stage}/stats.json.tmp", f"{stage}/stats.json")
print(json.dumps(dict(screenshots=tot, questions=out["questions"])))
