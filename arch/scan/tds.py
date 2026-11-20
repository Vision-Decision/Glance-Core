"""Exp 7a: does a cached clip carry the arrow of time? Temporal Divergence Score on Glance-Video clips (zero training).

For every positive Charades-STA test scan, the clips inside the ground-truth span are encoded forward and with their
frames reversed (same frames, same tokens, only the order changes). Questions (same block as the scan):
    "Does this clip show: <query>?"   and, for direction-paired actions, the opposite action
    (open<->close, take<->put, sit down<->stand up, turn on<->turn off, ...)
  TDS        mean |logit_fwd - logit_rev| of the query's yes-logit  (0 = the cached clip is time-symmetric for the model)
  AoT acc    directional queries: P(logit_fwd(query) > logit_rev(query))  (0.5 = chance)
  pair acc   directional queries: P( [l(q) - l(opp)]_fwd > [l(q) - l(opp)]_rev )
Controls: same statistics for non-directional queries ("holding", "sitting on", ...): TDS there measures the plain
sensitivity to frame order.  python -m arch.scan.tds --ckpt GV.pt --data /content/charades_pilot2 --out OUT.json
"""
import argparse, json, random, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from PIL import Image

from scan.pilot_charades import load_glance, frame_labels

PAIRS = [("open", "close"), ("opens", "closes"), ("opening", "closing"), ("take", "put"), ("takes", "puts"), ("taking", "putting"),
         ("sit down", "stand up"), ("sits down", "stands up"), ("sitting down", "standing up"),
         ("turn on", "turn off"), ("turns on", "turns off"), ("turning on", "turning off"),
         ("throw", "pick up"), ("throws", "picks up"), ("lie down", "get up"), ("lies down", "gets up"),
         ("dress", "undress"), ("puts on", "takes off"), ("put on", "take off"), ("walks in", "walks out"),
         ("awakens", "falls asleep"), ("start", "stop")]


def opposite(text):
    t = text.lower()
    for a, b in sorted(PAIRS, key=lambda p: -max(len(p[0]), len(p[1]))):
        for x, y in ((a, b), (b, a)):
            if re.search(rf"\b{re.escape(x)}\b", t):
                return re.sub(rf"\b{re.escape(x)}\b", y, t, count=1)
    return None


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--clip_vis", type=int, default=96)
    ap.add_argument("--max_clips", type=int, default=3, help="clips per scan (inside the span)")
    ap.add_argument("--max_scans", type=int, default=1200)
    a = ap.parse_args()
    meta = json.load(open(f"{a.data}/meta.json"))
    V, Q, fps = meta["videos"], meta["queries"], meta["fps"]
    G, cfg = load_glance(a.ckpt)
    G.dt, G.clip_max_vis = 1.0 / fps, a.clip_vis
    rng = random.Random(0)
    Q = [q for q in Q if q["vid"] in V]
    rng.shuffle(Q)
    rows = []
    for qi, q in enumerate(Q[: a.max_scans]):
        n = V[q["vid"]]["n_frames"]
        lab = frame_labels(n, fps, q["instances"], "clip", a.K, a.S)
        starts = [s for s, l in zip(range(0, max(1, n - a.K + 1), a.S), lab) if l]
        if not starts:
            continue
        starts = rng.sample(starts, min(a.max_clips, len(starts)))
        raw = [Image.open(f"{a.data}/frames/{q['vid']}/{k:03d}.jpg").convert("RGB") for k in range(n)]
        fw = [raw[s:s + a.K] if s + a.K <= n else raw[s:] + [raw[-1]] * (s + a.K - n) for s in starts]
        rv = [c[::-1] for c in fw]
        bank = G.encode(fw + rv, chunk=32)
        opp = opposite(q["text"])
        out = {}
        for tag, text in (("q", q["text"]), ("opp", opp)):
            if text is None:
                continue
            tl = G.query(bank, dict(type="noul", instr=f"Does this clip show: {text}?"))["typed_logits"]
            l = (tl[:, 1] - tl[:, 0]).float().numpy()
            out[tag] = (l[: len(fw)].tolist(), l[len(fw):].tolist())
        rows.append(dict(qid=q["qid"], vid=q["vid"], text=q["text"], opp=opp, cls=q["cls"], fwd=out["q"][0], rev=out["q"][1],
                         opp_fwd=out.get("opp", (None, None))[0], opp_rev=out.get("opp", (None, None))[1]))
        del bank
        if qi % 100 == 0:
            print(f"[tds] {qi} rows {len(rows)}", flush=True)
    # summary
    def stats(rs):
        d = np.array([f - r for x in rs for f, r in zip(x["fwd"], x["rev"])])
        if not len(d):
            return None
        rng2 = np.random.default_rng(0)
        bs = [np.mean(np.abs(d[rng2.integers(0, len(d), len(d))])) for _ in range(1000)]
        acc = (d > 0).astype(float)
        bsa = [acc[rng2.integers(0, len(acc), len(acc))].mean() for _ in range(1000)]
        o = dict(n_clips=len(d), n_scans=len(rs), TDS=float(np.abs(d).mean()), TDS_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 aot_acc=float(acc.mean()), aot_acc_ci95=[float(np.percentile(bsa, 2.5)), float(np.percentile(bsa, 97.5))],
                 mean_fwd_logit=float(np.mean([f for x in rs for f in x["fwd"]])))
        pr = [(np.array(x["fwd"]) - np.array(x["opp_fwd"])) - (np.array(x["rev"]) - np.array(x["opp_rev"])) for x in rs if x["opp"]]
        if pr:
            pr = np.concatenate(pr)
            o["pair_acc"] = float((pr > 0).mean())
            bsp = [(pr[rng2.integers(0, len(pr), len(pr))] > 0).mean() for _ in range(1000)]
            o["pair_acc_ci95"] = [float(np.percentile(bsp, 2.5)), float(np.percentile(bsp, 97.5))]
            o["pair_n"] = int(len(pr))
            o["fwd_prefers_query_over_opposite"] = float(np.mean(np.concatenate([np.array(x["fwd"]) > np.array(x["opp_fwd"]) for x in rs if x["opp"]])))
        return o
    S = dict(ckpt=a.ckpt, data=a.data, directional=stats([x for x in rows if x["opp"]]), non_directional=stats([x for x in rows if not x["opp"]]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(summary=S, rows=rows), open(a.out, "w"))
    print("[tds] DONE", json.dumps(S), flush=True)


if __name__ == "__main__":
    main()
