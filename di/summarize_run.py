"""Small summary of a finished run directory (run on the VM; the output is what gets pulled to di/results/).

    python -m di.summarize_run /content/out/di/text-glance-v1 --out /content/out/di/text_summary.json

Per-row wall times from results.jsonl (median / mean / p80 / p95, overall and per benchmark), status counts,
the >26-option rows (raw.chunked), plus the kit's own score files copied verbatim when present.
"""
import argparse
import collections
import json
import os
import statistics


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    per, status, chunked = collections.defaultdict(list), collections.Counter(), collections.Counter()
    allms = []
    with open(os.path.join(a.run_dir, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            b = r.get("dataset") or r.get("benchmark") or "?"
            status[r["status"]] += 1
            if r["status"] == "ok":
                per[b].append(r["total_wall_ms"]); allms.append(r["total_wall_ms"])
                if isinstance(r.get("raw_output"), dict) and r["raw_output"].get("chunked"):
                    chunked[b] += 1
    lat = lambda v: {"n": len(v), "median_ms": statistics.median(v), "mean_ms": statistics.fmean(v),
                     "p80_ms": q(v, 80), "p95_ms": q(v, 95), "max_ms": max(v)}
    res = {"run_dir": a.run_dir, "status": dict(status), "latency_all_rows": lat(allms),
           "latency_per_benchmark": {b: lat(v) for b, v in sorted(per.items())}, "chunked_rows": dict(chunked)}
    for f in ("scores.json", "index.json", "benchmark-summary.json", "vision-scores.json", "environment.json", "status.json"):
        p = os.path.join(a.run_dir, f)
        if os.path.exists(p):
            res[f] = json.load(open(p))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"status": res["status"], "latency_all_rows": res["latency_all_rows"]}, indent=1))


if __name__ == "__main__":
    main()
