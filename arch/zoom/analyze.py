"""Score the ScreenSpot-Pro zoom / reduced-prefix runs (arch/zoom/run_zoom.py rows). CPU.
python arch/zoom/analyze.py DIR --out outputs/arch/zoom/results.json
Per run: first-pass accuracy (acc1), in-block zoom (acc2), separate-prefill crop baseline (acc_rp, if run), the
presence-gated choice between full and zoom pass (zoom unless the crop's presence logit says absent: p2 < 0), paired
bootstrap CIs (1000) of acc2 - acc1 and acc2 - acc_rp, per-question latency components, by ui_type and group."""
import argparse, collections, glob, gzip, json, os
import numpy as np


def ci(x, n=1000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for f in sorted(glob.glob(f"{a.dir}/sspro_*.rows.jsonl.gz")):
        name = os.path.basename(f).replace(".rows.jsonl.gz", "")
        R = [json.loads(l) for l in gzip.open(f, "rt")]
        S = json.load(open(f.replace(".rows.jsonl.gz", ".json")))
        h1 = np.array([r["hit1"] for r in R], float)
        d = dict(n=len(R), prefix_vis=S["prefix_vis"], keep_ratio=S.get("keep_ratio", 1.0), layout=S["layout"],
                 crop_vis=S["crop_vis"], prefix_tok_median=S["prefix_tok_median"], ms_per_q=S["ms_per_q"],
                 acc1=float(h1.mean()), acc1_ci95=ci(h1))
        if "hit2" in R[0]:
            h2 = np.array([r["hit2"] for r in R], float)
            hg = np.array([r["hit2"] if r["p2"] > 0 else r["hit1"] for r in R], float)
            d.update(acc2=float(h2.mean()), acc2_ci95=ci(h2), delta_zoom=float((h2 - h1).mean()), delta_zoom_ci95=ci(h2 - h1),
                     acc_gated=float(hg.mean()), crop_tok_median=S.get("crop_tok_median"),
                     gt_in_crop=float(np.mean([r["crop"][0] <= (r["gt"][0] + r["gt"][2]) / 2 <= r["crop"][2] and
                                               r["crop"][1] <= (r["gt"][1] + r["gt"][3]) / 2 <= r["crop"][3] for r in R])))
            by = collections.defaultdict(lambda: [[], []])
            for r in R:
                for k in (r["meta"].get("ui_type"), r["meta"].get("group")):
                    by[k][0].append(r["hit1"]); by[k][1].append(r["hit2"])
            d["by"] = {k: dict(n=len(v[0]), acc1=float(np.mean(v[0])), acc2=float(np.mean(v[1]))) for k, v in by.items()}
            ms = S["ms_per_q"]
            d["latency_ms"] = dict(first_pass=ms["vit"] + ms["prefill"] + ms["stage1"], zoom_increment=ms["vit_crop"] + ms["stage2"],
                                   reprefill_increment=(ms["vit_crop"] + ms["reprefill"]) if ms.get("reprefill") else None)
        if "hit_rp" in R[0]:
            hr = np.array([r["hit_rp"] for r in R], float)
            d.update(acc_rp=float(hr.mean()), acc_rp_ci95=ci(hr), delta_inblock_minus_reprefill=float((h2 - hr).mean()),
                     delta_inblock_minus_reprefill_ci95=ci(h2 - hr),
                     agree_inblock_reprefill=float(np.mean(h2 == hr)))
        res[name] = d
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k not in ("by", "ms_per_q")}, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
