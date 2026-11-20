"""GUI-only training mixture for the sprint continuation (run on the VM after fetch_train.sh).

python evals/sprint/prep_train.py --out /content/data/prep_gui [--total_q 600000] [--desktop_vis 6400]

1. eval_hashes.json from ScreenSpot-v2 / ScreenSpot-Pro / VenusBench-GD images only (dHash, Hamming <= 3), so every
   training screenshot that near-duplicates an eval screenshot is dropped (same rule as dv/data/prep_v1.py).
2. dv.data.prep_v1.gui() over what fetch_train.sh downloaded (OS-Atlas desktop/mobile, AGUVIS web, GroundCUA)
   -> pool_gui.jsonl (+ calib_gui.jsonl); guisyn pool (to_v1.py) is hash-checked and appended.
3. mixture: per-domain visual-token caps / image caps, repeated up to max_repeat -> train.jsonl, stats.json.
"""
import argparse, collections, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dv.data import prep_v1 as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/content/data")
    ap.add_argument("--out", default="/content/data/prep_gui")
    ap.add_argument("--total_q", type=int, default=600000)
    ap.add_argument("--desktop_vis", type=int, default=6400)
    ap.add_argument("--mobile_vis", type=int, default=1280)
    ap.add_argument("--web_vis", type=int, default=1600)
    ap.add_argument("--img_cap", default="web_guienv=6000,mobile_uibert=1500,mobile_aw=4000",
                    help="image caps per sub-source; 'guisyn' caps all guisyn subs together")
    ap.add_argument("--neg", type=float, default=0.12)
    ap.add_argument("--max_repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    # 1. GUI eval hashes
    hp = f"{a.out}/eval_hashes.json"
    if not os.path.exists(hp):
        out = {}
        for name in ["ss2", "sspro", "venus"]:
            recs = [json.loads(l) for l in open(f"/content/gui/{name}.jsonl")]
            out[name] = [h for h in P._hash_many([os.path.join(r["image_root"], r["file"]) for r in recs]) if h]
        json.dump(out, open(hp, "w"))
        print({k: len(v) for k, v in out.items()}, flush=True)
    # 2. public GUI pools via prep_v1.gui()
    if not os.path.exists(f"{a.out}/pool_gui.jsonl"):
        ga = argparse.Namespace(raw=a.raw, out=a.out, seed=a.seed, gui_desktop=200000, gui_windows=0, gui_groundcua=2000000,
                                gui_jedi=0, gui_mobile=150000, gui_web=150000, gui_max_q=48, gui_neg=a.neg,
                                gui_max_vis=2048, gui_calib_per_dom=40)
        P.gui(ga)
    idx, _, _ = P.load_eval_index(argparse.Namespace(out=a.out))
    recs = [json.loads(l) for l in open(f"{a.out}/pool_gui.jsonl")]
    syn = [json.loads(l) for l in open(f"{a.out}/pool_guisyn.jsonl")] if os.path.exists(f"{a.out}/pool_guisyn.jsonl") else []
    hs = P._hash_many([f"{a.raw}/{r['img']}" for r in syn])
    syn_keep = [r for r, h in zip(syn, hs) if h and not idx.hit(h)]
    print("guisyn", len(syn), "kept", len(syn_keep), flush=True)
    recs += syn_keep
    # 3. mixture
    vis = dict(desktop=a.desktop_vis, mobile=a.mobile_vis, web=a.web_vis)
    cap = {k: int(v) for k, v in (kv.split("=") for kv in a.img_cap.split(",") if kv)}
    rng.shuffle(recs)
    seen, keep = collections.Counter(), []
    fam = lambda s: "guisyn" if "guisyn" in s else s
    for r in recs:
        dom = r["sub"].split("_")[0]
        f = fam(r["sub"])
        if f in cap and seen[f] >= cap[f]:
            continue
        seen[f] += 1
        r["max_vis"], r["upscale"] = vis[dom], 1.0
        r["plen"] = P.plen_image(*r["wh"], r["max_vis"], 1.0)
        keep.append(r)
    take, n, rep = [], 0, 0
    while n < a.total_q and rep < a.max_repeat:
        for r in keep:
            if n >= a.total_q:
                break
            take.append(r); n += len(r["questions"])
        rep += 1
    rng.shuffle(take)
    P.write_jsonl(f"{a.out}/train.jsonl", take)
    qs = [q for r in take for q in r["questions"]]
    st = dict(records=len(take), questions=len(qs), repeats=rep, absent=sum(not q["present"] for q in qs),
              point_only=sum(q["present"] and q.get("box") is None for q in qs),
              prefix_tokens=sum(r["plen"] for r in take),
              by_sub_q=collections.Counter(fam(r["sub"]) for r in take for _ in r["questions"]).most_common(),
              by_sub_img=collections.Counter(fam(r["sub"]) for r in take).most_common())
    json.dump(st, open(f"{a.out}/stats.json", "w"), indent=1)
    print(json.dumps(st, indent=1), flush=True)


if __name__ == "__main__":
    main()
