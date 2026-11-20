"""Row-level answer agreement between two results.jsonl files (our proxy for the maintainers' re-run check).

    python -m di.agreement runs/a/results.jsonl runs/b/results.jsonl

A row agrees when both runs are `ok` and every choice question has the same chosen key (noul: same side of 0.5);
rows only in one file, or not `ok` in both, are reported separately. Prints agreement, choice flips and max |dp|.
"""
import gzip
import json
import sys


def rows(path):
    op = gzip.open if path.endswith(".gz") else open
    out = {}
    with op(path, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["run_id"]] = r
    return out


def main(a, b):
    A, B = rows(a), rows(b)
    both = [k for k in A if k in B and A[k]["status"] == "ok" and B[k]["status"] == "ok"]
    agree, flips, dmax = 0, 0, 0.0
    for k in both:
        x, y = A[k]["response"]["answers"], B[k]["response"]["answers"]
        same = True
        for q in x:
            if x[q]["type"] == "choice":
                same &= x[q]["choice"] == y[q]["choice"]
                dmax = max(dmax, max(abs(p - y[q]["probabilities"][o]) for o, p in x[q]["probabilities"].items()))
            else:
                same &= (x[q]["noul"] >= 0.5) == (y[q]["noul"] >= 0.5)
                dmax = max(dmax, abs(x[q]["noul"] - y[q]["noul"]))
        agree += same
        flips += not same
    res = {"rows_a": len(A), "rows_b": len(B), "compared": len(both), "agreement": agree / max(1, len(both)),
           "rows_with_flip": flips, "max_abs_dp": dmax,
           "status_mismatch": sum(1 for k in A if k in B and A[k]["status"] != B[k]["status"]),
           "target": ">= 0.985 (bf16; brief's figure, not found in the public kit/methodology)"}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
