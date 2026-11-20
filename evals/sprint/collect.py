"""Collect the sprint analysis files on the VM into one compact JSON for outputs/sprint_gui/.

python evals/sprint/collect.py --dir /content/sprint --out /content/sprint/sprint_gui_results.json
Keeps, per run: run info (ckpt, resolution, wall time), every decoder's overall accuracy, the default ('box') decoder's
per-group accuracies, VenusBench refusal / official overall, and zoom / multi-candidate zoom results.
"""
import argparse, glob, json, os


def slim(score):
    return {k: v["acc"] for k, v in score.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/content/sprint")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = {}
    for f in glob.glob(f"{a.dir}/*.info.json"):
        d = json.load(open(f))
        info[os.path.basename(d["out"])] = {k: d[k] for k in ("ckpt", "max_vis", "upscale", "zoom_from", "zoom_frac", "wall_s", "n", "vis_tokens_median") if k in d}
    res = dict(runs={}, zoom={}, mzoom={})
    for f in sorted(glob.glob(f"{a.dir}/an_*.json")):
        d = json.load(open(f))
        for k, v in d.items():
            if k.startswith("zoom:"):
                res["zoom"][k[5:]] = dict(decoders={n: s["all"]["acc"] for n, s in v["decoders"].items()} if "decoders" in v else None,
                                          box_groups=slim(v["zoom"]), info=info.get(k[5:]))
            elif k.startswith("mzoom:"):
                res["mzoom"][k[6:]] = dict(rules={n: s["all"]["acc"] for n, s in v.items()},
                                           presence_groups=slim(v["presence"]), info=info.get(k[6:]))
            elif k.startswith("ens:"):
                res.setdefault("ens", {})[k[4:]] = slim(v["fused"])
            else:
                res["runs"][k] = dict(decoders={n: s["all"]["acc"] for n, s in v["decoders"].items()},
                                      box_groups=slim(v["decoders"]["box"]), refusal_acc=v.get("refusal_acc"),
                                      venus_official_overall=v.get("venus_official_overall"), info=info.get(k))
    json.dump(res, open(a.out, "w"), indent=1)
    print(len(res["runs"]), "runs", len(res["zoom"]), "zoom", len(res["mzoom"]), "mzoom")


if __name__ == "__main__":
    main()
