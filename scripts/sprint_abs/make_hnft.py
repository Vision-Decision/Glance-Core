"""Hard-negative fine-tune mixture (VM): continue v1 on natural replay + guisyn (verified hard negatives), keeping every
guisyn image used for calibration (guisyn_calib.jsonl) and every eval image out.

python scripts/sprint_abs/make_hnft.py --pool_natural /content/sabs/data/pool_natural.jsonl \
   --guisyn_shards /content/guisyn --calib /content/sabs/data/guisyn_calib.jsonl --out /content/sabs/data/hnft.jsonl
Natural: --n_nat records sampled from v1's own natural pool (same images as v1 training; 28% absent, 65% hard).
GUI: guisyn records (not calibration images), positives + negatives with negatives up to --gui_neg of questions,
hard (same-app, verified absent) preferred."""
import argparse, glob, json, os, random, tarfile, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dv.data.prep_v1 import plen_image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool_natural", required=True); ap.add_argument("--guisyn_shards", required=True)
ap.add_argument("--calib", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--data_root", default="/content/data")
ap.add_argument("--n_nat", type=int, default=6000); ap.add_argument("--n_gui", type=int, default=3000)
ap.add_argument("--gui_neg", type=float, default=0.35); ap.add_argument("--max_q", type=int, default=32)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
rng = random.Random(a.seed)
calib_imgs = {json.loads(l)["img"] for l in open(a.calib)}
nat = [json.loads(l) for l in open(a.pool_natural)]
rng.shuffle(nat)
recs = nat[: a.n_nat]
gui, need = [], {}
for src in ("desktop", "web"):
    for jf in sorted(glob.glob(f"{a.guisyn_shards}/shards/{src}/*.jsonl")):
        for l in open(jf):
            r = json.loads(l)
            img = f"guisyn/{r['image']}"
            if img in calib_imgs:
                continue
            gui.append((jf, r))
rng.shuffle(gui)
for jf, r in gui[: a.n_gui]:
    pos = [q for q in r["questions"] if q["present"]]
    hard = [q for q in r["questions"] if q["kind"] == "negative_hard"]
    easy = [q for q in r["questions"] if q["kind"] == "negative_easy"]
    rng.shuffle(pos); rng.shuffle(hard); rng.shuffle(easy)
    n_neg = round(a.max_q * a.gui_neg)
    neg = (hard + easy)[:n_neg]
    qs = pos[: a.max_q - len(neg)] + neg
    if not qs:
        continue
    rng.shuffle(qs)
    W, H = r["W"], r["H"]
    recs.append(dict(src="gui", sub=("desktop_guisyn_" + r["app"]) if r["source"] == "desktop" else "web_guisyn",
                     img=f"guisyn/{r['image']}", wh=[W, H], max_vis=2048, plen=plen_image(W, H, 2048),
                     questions=[dict(type="box", text=q["text"], present=bool(q["present"]),
                                     box=[float(v) for v in q["box"]] if q["present"] else None,
                                     neg={"negative_hard": "hard", "negative_easy": "easy"}.get(q["kind"])) for q in qs]))
    need.setdefault(jf, set()).add(r["image"])
for jf, names in need.items():
    with tarfile.open(jf[:-len(".jsonl")] + ".images.tar") as t:
        t.extractall(f"{a.data_root}/guisyn", members=[m for m in t.getmembers() if m.name in names], filter="data")
rng.shuffle(recs)
with open(a.out, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
qs = [q for r in recs for q in r["questions"]]
print(json.dumps(dict(records=len(recs), questions=len(qs), absent=sum(not q["present"] for q in qs),
                      hard=sum(q.get("neg") == "hard" for q in qs), gui_records=sum(r["src"] == "gui" for r in recs))))
