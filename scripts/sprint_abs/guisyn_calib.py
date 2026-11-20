"""guisyn-v0 -> GUI calibration records for the sprint (VM). guisyn is NOT in the v1 / v1.1 training mixtures and
contains no ScreenSpot / VenusBench image, so it is a held-out, non-eval GUI source with *verified* negatives.

python scripts/sprint_abs/guisyn_calib.py --shards /content/guisyn --data_root /content/data --out /content/sabs/data/guisyn_calib.jsonl
Per app (desktop) / per site group (web): up to --per_app screenshots, up to --max_q questions each, negatives capped at
--neg_frac. Question meta keeps kind (name/function/position/icon/negative_hard/negative_easy) and app; the offline
analysis splits fit/test by app (cross-app held-out check)."""
import argparse, glob, json, os, random, tarfile, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True); ap.add_argument("--data_root", default="/content/data")
ap.add_argument("--out", required=True); ap.add_argument("--per_app", type=int, default=40)
ap.add_argument("--max_q", type=int, default=24); ap.add_argument("--neg_frac", type=float, default=0.35)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
by_app = {}
for src in ("desktop", "web"):
    for jf in sorted(glob.glob(f"{a.shards}/shards/{src}/*.jsonl")):
        for l in open(jf):
            r = json.loads(l)
            app = r["app"] if src == "desktop" else "web:" + (r.get("url") or r["image"]).split("/")[2 if "//" in (r.get("url") or "") else 0]
            by_app.setdefault(app, []).append((jf, r))
need = {}
recs = []
for app, items in sorted(by_app.items()):
    rng.shuffle(items)
    lim = a.per_app if not app.startswith("web:") else 2
    for jf, r in items[:lim]:
        pos = [q for q in r["questions"] if q["present"]]
        neg = [q for q in r["questions"] if not q["present"]]
        rng.shuffle(pos); rng.shuffle(neg)
        n_neg = min(len(neg), round(a.max_q * a.neg_frac))
        qs = pos[: a.max_q - n_neg] + neg[:n_neg]
        if not qs:
            continue
        rng.shuffle(qs)
        out_q = [dict(type="box", text=q["text"], present=bool(q["present"]),
                      box=[float(v) for v in q["box"]] if q["present"] else None,
                      neg={"negative_hard": "hard", "negative_easy": "easy"}.get(q["kind"]),
                      meta=dict(kind=q["kind"], app=app, source=r["source"])) for q in qs]
        recs.append(dict(src="gui", sub=("desktop_guisyn_" + app) if r["source"] == "desktop" else "web_guisyn",
                         img=f"guisyn/{r['image']}", wh=[r["W"], r["H"]], max_vis=2048, plen=0, questions=out_q))
        need.setdefault(jf, set()).add(r["image"])
# extract only the needed images
for jf, names in need.items():
    with tarfile.open(jf[:-len(".jsonl")] + ".images.tar") as t:
        mem = [m for m in t.getmembers() if m.name in names]
        t.extractall(f"{a.data_root}/guisyn", members=mem, filter="data")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
web = sum(1 for k in by_app if k.startswith("web:"))
print(json.dumps(dict(records=len(recs), apps=len(by_app) - web, web_sites=web,
                      questions=sum(len(r["questions"]) for r in recs),
                      absent=sum(not q["present"] for r in recs for q in r["questions"]))))
