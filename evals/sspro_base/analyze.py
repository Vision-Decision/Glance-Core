"""Compare base Qwen3.5-4B on ScreenSpot-Pro at native resolution (cap 16,384 visual tokens, no upscale) with the
earlier <=4,000-token run, on the same items; reconcile the 56.0 vs 55.6 base numbers.

python evals/sspro_base/analyze.py   (from the repo root; writes outputs/sspro_base/results.json)
"""
import json, math, random, statistics, subprocess
from collections import defaultdict

NAT = "outputs/sspro_base/base_sspro_16384u1.jsonl"
OLD = "outputs/eval_v0_xdomain/rows/base_sspro_4000.jsonl"
OUT = "outputs/sspro_base/results.json"
GLANCE = "outputs/sprint_gui/sprint_gui_results.json"


def load(lines):
    return {r["id"]: r for r in (json.loads(l) for l in lines if l.strip())}


def acc(rows):
    return round(100 * sum(bool(r["hit"]) for r in rows) / len(rows), 2) if rows else None


def vis(r):
    return (r["in_wh"][0] // 32) * (r["in_wh"][1] // 32)


def main():
    nat = load(open(NAT))
    old = load(open(OLD))
    old718 = load(subprocess.check_output(["git", "show", f"dc66d06:{OLD}"]).decode().splitlines())
    ids = sorted(nat)
    assert set(ids) <= set(old)
    a = [bool(nat[i]["hit"]) for i in ids]
    b = [bool(old[i]["hit"]) for i in ids]
    n01 = sum(x and not y for x, y in zip(a, b))  # native right, 4000 wrong
    n10 = sum(y and not x for x, y in zip(a, b))
    # exact two-sided McNemar (binomial on discordant pairs)
    k, m = min(n01, n10), n01 + n10
    p = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
    rng = random.Random(0)
    diffs = []
    for _ in range(5000):
        s = [rng.randrange(len(ids)) for _ in ids]
        diffs.append(100 * sum(a[j] - b[j] for j in s) / len(s))
    diffs.sort()
    groups = defaultdict(list)
    for i in ids:
        mt = nat[i]["meta"]
        for g in (f"group/{mt['group']}", f"type/{mt['ui_type']}"):
            groups[g].append(i)
    by = {g: dict(n=len(v), native=acc([nat[i] for i in v]), cap4000=acc([old[i] for i in v]))
          for g, v in sorted(groups.items())}
    glance = json.load(open(GLANCE))["runs"]["v11f_sspro_16384u1"]
    res = dict(
        setting=dict(model="Qwen/Qwen3.5-4B (generative, greedy, enable_thinking=False)",
                     prompt="official ScreenSpot-Pro Qwen3.5 computer_use click prompt, assistant pre-filled to coordinate",
                     resize="resize_dims(W,H,16384,max_upscale=1.0), 32px per visual token (same rule as Glance v11f_sspro_16384u1)",
                     order="seeded random order (seed 0), run cut by a time limit -> random subset" if len(ids) < len(old) else "all items",
                     scoring="hit = predicted point inside GT box (same as base_sspro_4000)"),
        n=len(ids), n_total=len(old),
        native_16384u1=acc([nat[i] for i in ids]),
        cap4000_same_items=acc([old[i] for i in ids]),
        cap4000_all_1581=acc(list(old.values())),
        diff_native_minus_4000=round(100 * (sum(a) - sum(b)) / len(ids), 2),
        diff_bootstrap95=[round(diffs[125], 2), round(diffs[-126], 2)],
        mcnemar=dict(native_only=n01, cap4000_only=n10, p_exact=round(p, 4)),
        vis_tokens_median=dict(native=statistics.median(vis(nat[i]) for i in ids),
                               cap4000=statistics.median(vis(old[i]) for i in ids)),
        refused_native=sum(bool(nat[i]["refused"]) for i in ids),
        ms_median_native=round(statistics.median(nat[i]["ms"] for i in ids), 1),
        by_group=by,
        glance_v11f_native=dict(acc=round(100 * glance["box_groups"]["all"], 2), n=glance["info"]["n"],
                                vis_tokens_median=glance["info"]["vis_tokens_median"]),
        reconcile=dict(
            dc66d06_rows=len(old718), dc66d06_acc=acc(list(old718.values())),
            f804073_rows=len(old), f804073_acc=acc(list(old.values())),
            first718_identical_hits=all(old718[i]["hit"] == old[i]["hit"] for i in old718)),
    )
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "by_group"}, indent=1))
    for g, v in by.items():
        print(g, v)


if __name__ == "__main__":
    main()
