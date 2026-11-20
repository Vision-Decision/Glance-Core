"""guisyn shards -> decision-vision v1 GUI pool records (same schema as dv/data/prep_v1.py gui()).

  python scripts/guisyn/to_v1.py --shards /content/guisyn-v0 --data_root /content/data --out /content/data/prep_v1 \
      [--extract] [--max_q 48] [--calib_per_sub 60] [--sources desktop,web]

Input: a local snapshot of the HF dataset (hf download Vision-Decision/guisyn-v0 --repo-type dataset --local-dir ...).
--extract unpacks every {shard}.images.tar into {data_root}/guisyn/ (members are images/{source}/{id}.webp).
Output: {out}/pool_guisyn.jsonl + calib_guisyn.jsonl, records like
  {"src": "gui", "sub": "desktop_guisyn_<app>" | "web_guisyn", "img": "guisyn/images/...webp", "wh": [W, H],
   "max_vis": 2048, "plen": ..., "questions": [{"type": "box", "text", "present", "box" px | null, "neg": "hard"|"easy"|null}]}
`sub` starts with desktop_/web_ so prep_v1 mix() applies its per-domain --gui_vis / --gui_img_cap settings.
To use it in the mixture: cat pool_guisyn.jsonl >> pool_gui.jsonl (or point mix at it) before `prep_v1 mix`.
"""
import argparse, glob, json, os, random, sys, tarfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dv.data.prep_v1 import plen_image  # noqa: E402

NEG = {"negative_hard": "hard", "negative_easy": "easy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--data_root", default="/content/data")
    ap.add_argument("--out", default="/content/data/prep_v1")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--sources", default="desktop,web")
    ap.add_argument("--max_q", type=int, default=48)
    ap.add_argument("--max_vis", type=int, default=2048)
    ap.add_argument("--neg_frac", type=float, default=0.2, help="cap on absent questions per record after subsampling")
    ap.add_argument("--calib_per_sub", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    recs, calib, n_sub = [], [], {}
    for src in a.sources.split(","):
        for jf in sorted(glob.glob(f"{a.shards}/shards/{src}/*.jsonl")):
            if a.extract:
                with tarfile.open(jf[:-len(".jsonl")] + ".images.tar") as t:
                    t.extractall(f"{a.data_root}/guisyn", filter="data")
            for l in open(jf):
                r = json.loads(l)
                pos = [q for q in r["questions"] if q["present"]]
                neg = [q for q in r["questions"] if not q["present"]]
                rng.shuffle(pos); rng.shuffle(neg)
                n_neg = min(len(neg), round(a.max_q * a.neg_frac))
                qs = pos[: a.max_q - n_neg] + neg[:n_neg]
                rng.shuffle(qs)
                out_q = [dict(type="box", text=q["text"], present=bool(q["present"]),
                              box=[float(v) for v in q["box"]] if q["present"] else None,
                              neg=NEG.get(q["kind"])) for q in qs]
                sub = f"desktop_guisyn_{r['app']}" if r["source"] == "desktop" else "web_guisyn"
                W, H = r["W"], r["H"]
                rec = dict(src="gui", sub=sub, img=f"guisyn/{r['image']}", wh=[W, H], max_vis=a.max_vis,
                           plen=plen_image(W, H, a.max_vis), questions=out_q)
                if n_sub.get(sub, 0) < a.calib_per_sub:
                    calib.append(rec)
                else:
                    recs.append(rec)
                n_sub[sub] = n_sub.get(sub, 0) + 1
    rng.shuffle(recs)
    for name, rs in (("pool_guisyn", recs), ("calib_guisyn", calib)):
        with open(f"{a.out}/{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nq = sum(len(r["questions"]) for r in recs)
    print(json.dumps(dict(records=len(recs), calib=len(calib), questions=nq,
                          absent=sum(not q["present"] for r in recs for q in r["questions"]), subs=len(n_sub))))


if __name__ == "__main__":
    main()
