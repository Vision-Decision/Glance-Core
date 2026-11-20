"""Latency pre-flight that mimics the board's latency-v1 protocol: 750 timed rows + 10 warm-up rows, a systematic
sample (random start, seed 20260926) over the scope sorted by benchmark then payload length, one process, one
request at a time, CUDA-synchronized wall time per row (request in -> all answers out).

    PYTHONPATH=. python -m di.latency_probe --suite /content/di/suite-0.3 --out /content/out/di/latency_probe.json

The board samples its own (private) rows; this is a guide. Gate: median, mean and p80 must each be <= 1000 ms.
"""
import argparse
import collections
import json
import os
import random
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def sample_rows(suite_dir, edition, n, warm, seed=20260926):
    from decision_index.suite.io import Suite, read_jsonl
    suite = Suite(suite_dir, edition)
    keep = suite.in_edition
    rows = []
    for path in suite.row_paths:
        for row in read_jsonl(path):
            e = row["_evaluation"]
            if keep(e):
                rows.append((e.get("dataset") or "", len(json.dumps({"state": row["state"], "questions": row["questions"]})), row))
    rows.sort(key=lambda r: (r[0], r[1]))
    step = len(rows) / (n + warm)
    start = random.Random(seed).random() * step
    pick = [rows[int(start + i * step)] for i in range(n + warm)]
    random.Random(seed + 1).shuffle(pick)
    return [p[2] for p in pick[:warm]], [p[2] for p in pick[warm:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--edition", default="0.3")
    ap.add_argument("--n", type=int, default=750)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--option", action="append", default=[], help="GlanceEngine option key=value")
    ap.add_argument("--out", default="latency_probe.json")
    a = ap.parse_args()
    import torch
    from di.engine import GlanceEngine, Unsupported
    eng = GlanceEngine(**dict(o.split("=", 1) for o in a.option))
    warm, timed = sample_rows(a.suite, a.edition, a.n, a.warm)
    for r in warm:
        try:
            eng(r["state"], r["questions"])
        except Unsupported:
            pass
    per = collections.defaultdict(list)
    ms, unsup = [], 0
    for r in timed:
        eng.synchronize(); t = time.perf_counter()
        try:
            eng(r["state"], r["questions"])
        except Unsupported:
            unsup += 1
            continue
        eng.synchronize()
        x = (time.perf_counter() - t) * 1000
        ms.append(x); per[r["_evaluation"].get("dataset")].append(x)
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]
    res = {"n": len(ms), "unsupported": unsup, "median_ms": statistics.median(ms), "mean_ms": statistics.fmean(ms),
           "p80_ms": q(ms, 80), "p95_ms": q(ms, 95), "max_ms": max(ms),
           "gate_ok": statistics.median(ms) <= 1000 and statistics.fmean(ms) <= 1000 and q(ms, 80) <= 1000,
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
           "per_benchmark_median_ms": {b: round(statistics.median(v), 1) for b, v in sorted(per.items())}}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
