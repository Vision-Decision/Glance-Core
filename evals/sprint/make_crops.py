"""Zoom-scale training crops from the GUI pools (sprint ft2): teaches the model the element scale it sees in the
crop-and-zoom second prefill (a 0.2-0.5 crop of a desktop screenshot, upscaled up to 2x), and adds verified
"not in this crop" negatives for the presence gate.

python evals/sprint/make_crops.py --pool /content/data/prep_gui_v2/pool_gui.jsonl,/content/data/prep_gui_v2/pool_guisyn.jsonl \
    --replay /content/data/prep_gui_v2/train.jsonl --out /content/data/prep_crops

Per source record (desktop subs with pixel boxes): 1-2 crops of frac ~ U(0.2, 0.5) of each side, centred near a random
element. Present questions: elements with >= 90% of their box inside the crop (box clipped, crop-relative px).
Absent questions: elements entirely outside the crop whose text matches no element text inside it (case-insensitive),
capped at abs_frac of the record. Records carry crop=[x1,y1,x2,y2] (full-image px), max_vis 4096, upscale 2 -- the
same preprocessing as the zoom pass. train.jsonl = crops + a replay sample of the ft1 mixture.
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dv.data import prep_v1 as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--replay", default="")
    ap.add_argument("--replay_frac", type=float, default=0.35, help="replay records as a fraction of crop records")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_crops", type=int, default=9000)
    ap.add_argument("--abs_frac", type=float, default=0.25)
    ap.add_argument("--max_q", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(l) for p in a.pool.split(",") for l in open(p)]
    recs = [r for r in recs if r["sub"].startswith("desktop")]
    rng.shuffle(recs)
    out, st = [], dict(src_records=len(recs), crops=0, present=0, absent=0)
    for r in recs:
        if len(out) >= a.max_crops:
            break
        W, H = r["wh"]
        qs = [q for q in r["questions"] if q.get("present") and q.get("box")]
        if len(qs) < 4:
            continue
        for _ in range(rng.choice([1, 2])):
            fx, fy = rng.uniform(0.2, 0.5), rng.uniform(0.2, 0.5)
            cw, ch = W * fx, H * fy
            e = rng.choice(qs)["box"]
            cx = (e[0] + e[2]) / 2 + rng.uniform(-0.35, 0.35) * cw
            cy = (e[1] + e[3]) / 2 + rng.uniform(-0.35, 0.35) * ch
            x1 = min(max(0, cx - cw / 2), W - cw); y1 = min(max(0, cy - ch / 2), H - ch)
            x2, y2 = x1 + cw, y1 + ch
            pres, outside = [], []
            for q in qs:
                b = q["box"]
                area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
                ix = max(0, min(b[2], x2) - max(b[0], x1)) * max(0, min(b[3], y2) - max(b[1], y1))
                if ix / area >= 0.9:
                    nb = [max(b[0], x1) - x1, max(b[1], y1) - y1, min(b[2], x2) - x1, min(b[3], y2) - y1]
                    if nb[2] - nb[0] >= 1 and nb[3] - nb[1] >= 1:
                        pres.append(dict(type="box", text=q["text"], present=True, box=[round(v, 1) for v in nb]))
                elif ix == 0:
                    outside.append(q["text"])
            if len(pres) < 2:
                continue
            inside_txt = {p["text"].lower() for p in pres}
            # partially-overlapping elements are neither positives nor negatives
            neg = [t for t in dict.fromkeys(outside) if t.lower() not in inside_txt]
            rng.shuffle(pres); rng.shuffle(neg)
            n_pos = min(len(pres), a.max_q - round(a.max_q * a.abs_frac))
            n_neg = min(len(neg), round(n_pos * a.abs_frac / (1 - a.abs_frac)))
            qq = pres[:n_pos] + [dict(type="box", text=t, present=False, box=None, neg="crop") for t in neg[:n_neg]]
            rng.shuffle(qq)
            w, h = P.resize_dims(cw, ch, 4096, 2.0)
            out.append(dict(src="gui", sub=r["sub"] + "_crop", img=r["img"], wh=[W, H], crop=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            max_vis=4096, upscale=2.0, plen=(w // 32) * (h // 32) + 16, questions=qq))
            st["crops"] += 1; st["present"] += n_pos; st["absent"] += n_neg
    P.write_jsonl(f"{a.out}/pool_crops.jsonl", out)
    train = list(out)
    if a.replay:
        rep = [json.loads(l) for l in open(a.replay)]
        rng.shuffle(rep)
        train += rep[: int(len(out) * a.replay_frac)]
        st["replay"] = int(len(out) * a.replay_frac)
    rng.shuffle(train)
    P.write_jsonl(f"{a.out}/train.jsonl", train)
    st["train_records"] = len(train)
    st["train_questions"] = sum(len(r["questions"]) for r in train)
    st["prefix_tokens"] = sum(r["plen"] for r in train)
    json.dump(st, open(f"{a.out}/stats.json", "w"), indent=1)
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
