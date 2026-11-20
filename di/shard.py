"""Split the 0.3 text suite across VMs and merge the results (each row is still timed alone on one G4).

    python -m di.shard make  --suite /content/di/suite-0.3 --k 0 --cuts 0,11,20 --out /content/di/shard0.jsonl.gz
    python -m di.shard strip results.jsonl out.jsonl.gz      # drop `payload` (suite text) before moving result rows
    python -m di.shard merge a.jsonl[.gz] b.jsonl[.gz] --out merged/results.jsonl

Shard of a row = sha256(run_id) mod 20, assigned by `--cuts` (shard k gets [cuts[k], cuts[k+1])). Only rows of the
edition (read-time filter, as `pipeline` applies) are written; linked cases are kept whole (group_id hash).
"""
import argparse
import gzip
import hashlib
import json


def opener(p, mode="rt"):
    return gzip.open(p, mode) if p.endswith(".gz") else open(p, mode)


def make(a):
    from decision_index.suite.io import Suite, read_jsonl
    suite = Suite(a.suite, "0.3")
    keep = suite.in_edition
    cuts = [int(x) for x in a.cuts.split(",")]
    lo, hi = cuts[a.k], cuts[a.k + 1]
    n = 0
    with gzip.open(a.out, "wt") as f:
        for path in suite.row_paths:
            for row in read_jsonl(path):
                e = row["_evaluation"]
                if not keep(e):
                    continue
                h = int(hashlib.sha256(str(e.get("group_id") or e["run_id"]).encode()).hexdigest(), 16) % 20
                if lo <= h < hi:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
    print(json.dumps({"shard": a.k, "rows": n, "buckets": [lo, hi]}))


def strip(a):
    n = 0
    with opener(a.src) as f, gzip.open(a.dst, "wt") as g:
        for line in f:
            if line.strip():
                r = json.loads(line); r.pop("payload", None); g.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print(json.dumps({"rows": n, "out": a.dst}))


def merge(a):
    seen, n = set(), 0
    with open(a.out, "w") as g:
        for p in a.inputs:
            with opener(p) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["run_id"] in seen:
                        continue
                    seen.add(r["run_id"]); g.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print(json.dumps({"rows": n, "out": a.out}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--suite", required=True); m.add_argument("--k", type=int, required=True)
    m.add_argument("--cuts", default="0,11,20"); m.add_argument("--out", required=True)
    s = sub.add_parser("strip"); s.add_argument("src"); s.add_argument("dst")
    g = sub.add_parser("merge"); g.add_argument("inputs", nargs="+"); g.add_argument("--out", required=True)
    a = ap.parse_args()
    {"make": make, "strip": strip, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()
