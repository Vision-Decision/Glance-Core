"""Exact MAP decision for the trained AR heads (audit Q1): best-first search pruned by q(none).

python -m ablation.exactmap_ab --ckpt CKPT --out OUT.json --rows ROWS.json [--image_stride 1] [--max_nodes 4000]

For each question, MAP refuses iff  log q("none" <end>)  >=  max over grammar-valid box sequences s of log q(s)
(ties -> refuse, as in the main analysis). Scores are full-vocabulary log-probs (the same quantities as eval_ab's lp_none /
lp_box and beam_ab). Search: best-first (uniform-cost) over answer prefixes ordered by prefix log-prob; a prefix
is pushed only if its log-prob > L_none = log q(none <end>) (prefix log-probs never increase, so nothing pruned can
beat "none"). The first *complete* box sequence popped is the exact MAP box (every other queued item has lower
log-prob, and their extensions too) -> accept. An empty queue -> exact MAP is "none" -> refuse.
Early accept: the refusal DECISION only needs one complete box with log-prob > L_none, so a question is accepted as
soon as such a box is generated (status "box"; lp_map is then a lower bound of the MAP box log-prob and map_box need
not be the arg-max box).
Nodes are expanded in batches (per image, up to --pop_k nodes per question per round, stopping before a complete
node so the order stays exact); each round is ONE block forward of all popped prefixes against the image's prefix
cache (net.block re-runs the question block + prefix tokens, exactly like eval_ab.ar_decode, so no per-node
recurrent state is stored). Questions that exceed --max_nodes expansions are reported as undecided, with the best
complete box found (lower bound) -> refusal is bracketed.
Eval set as beam_ab: gRefCOCO-val-merged minus multi-target (RefCOCO val present 10,834 + no-target 8,905).
"""
import argparse, heapq, json, time
import numpy as np
import torch

from ablation.common import ABConfig, ARDataset, collate_ar, build_ar, ar_load
from ablation.eval_ab import iou_np, _pad_blocks
from ablation.train_ab import to_dev


@torch.no_grad()
def block_logp(net, pc, seqs, starts, q_img, q_split, pad, dev, chunk=384):
    out = []
    for i in range(0, len(seqs), chunk):
        ids, valid, pos = _pad_blocks(seqs[i:i + chunk], starts[i:i + chunk], dev, pad)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hq = net.block(pc, ids, valid, pos, q_img[i:i + chunk], q_split[i:i + chunk])
            last = valid.sum(1) - 1
            lg = net.logits(hq[torch.arange(len(ids), device=dev), last]).float()
        out.append(lg.log_softmax(-1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", default="grefcoco_val_merged")
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--image_stride", type=int, default=1)
    ap.add_argument("--limit_images", type=int, default=0)
    ap.add_argument("--max_nodes", type=int, default=4000)
    ap.add_argument("--pop_k", type=int, default=24)
    ap.add_argument("--split_k", type=int, default=1,
                    help="refinement test (Prop. 2): every coordinate token split into k equal sub-forms (no retraining); "
                         "a box of n coordinate tokens then has log q - n log k. Atomic (n=4) and hier (n=8) only.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", required=True)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ABConfig().update({k: v for k, v in sd["cfg"].items() if k in ABConfig.__dataclass_fields__})
    net, fmt, proc = build_ar(cfg, "cuda")
    ar_load(a.ckpt, net)
    net.eval()
    ds = ARDataset(f"{a.prep_dir}/eval_{a.set}.jsonl", cfg.image_root, proc, cfg, train=False, fmt=fmt)
    recs = []
    for r in ds.recs:
        keep = [q for q in r["questions"] if not (q["present"] and q.get("n_targets", 1) > 1)]
        if keep:
            recs.append(dict(r, questions=keep))
    recs = recs[:: a.image_stride]
    if a.limit_images:
        recs = recs[: a.limit_images]
    ds.recs = recs
    tok = ds.tok
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=6,
                                     collate_fn=lambda it: collate_ar(it, tok, with_answer=False))
    allowed_memo = {}
    n_coord = 0
    if a.split_k > 1:
        assert cfg.variant in ("ar_atomic", "ar_hier"), "split_k: fixed-length formats only"
        n_coord = 4 if cfg.variant == "ar_atomic" else 8

    def allowed(g):
        k = tuple(g)
        if k not in allowed_memo:
            allowed_memo[k] = [t for t in fmt.allowed(list(g)) if not (len(g) == 0 and t == fmt.none)]
        return allowed_memo[k]

    rows, t0, n_fwd = [], time.time(), 0
    for bi, batch in enumerate(dl):
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        dev = b["q_ids"].device
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = net.image_features(b["pixel_values"], b["grid_thw"])
            pc = net.prefill(b, feats)
        M = b["q_ids"].shape[0]
        q_tok = [b["q_ids"][m, : int(b["q_valid"][m].sum())].tolist() for m in range(M)]
        starts = b["q_pos"][0, :, 0].tolist()
        q_img, q_split = b["q_img"], b["q_last"]
        # root logits + "none" continuation
        lp0 = block_logp(net, pc, q_tok + [q + [fmt.none] for q in q_tok], starts + starts,
                         torch.cat([q_img, q_img]), torch.cat([q_split, q_split]), tok.pad, dev)
        n_fwd += 1
        L_none = [float(lp0[m, fmt.none] + lp0[M + m, fmt.box_end]) + n_coord * np.log(a.split_k) for m in range(M)]
        heaps, status, best, nexp = [[] for _ in range(M)], [None] * M, [None] * M, [0] * M
        cnt = 0
        for m in range(M):
            lr = lp0[m]
            for t in allowed([]):
                v = float(lr[t])
                if v > L_none[m]:
                    cnt += 1
                    heapq.heappush(heaps[m], (-v, cnt, (t,), t in fmt.stop))
            if not heaps[m]:
                status[m] = "none"
        while True:
            pops = []
            for m in range(M):
                if status[m] is not None:
                    continue
                h = heaps[m]
                if not h:
                    status[m] = "none"; continue
                if h[0][3]:                      # top of queue is a complete box: exact MAP box
                    status[m] = "box"; best[m] = (-h[0][0], list(h[0][2])); continue
                if nexp[m] >= a.max_nodes:
                    status[m] = "undecided"; continue
                k = 0
                while h and not h[0][3] and k < a.pop_k:
                    nv, _, g, _ = heapq.heappop(h)
                    pops.append((m, -nv, g)); k += 1
                nexp[m] += k
            if not pops:
                break
            seqs = [q_tok[m] + list(g) for m, _, g in pops]
            mi = torch.tensor([m for m, _, _ in pops], device=dev)
            lps = block_logp(net, pc, seqs, [starts[m] for m, _, _ in pops], q_img[mi], q_split[mi], tok.pad, dev)
            n_fwd += 1
            for (m, v, g), lr in zip(pops, lps):
                if status[m] is not None:
                    continue
                al = allowed(g)
                vals = (lr[al] + v).tolist()
                for t, nv in zip(al, vals):
                    if nv > L_none[m]:
                        cnt += 1
                        heapq.heappush(heaps[m], (-nv, cnt, g + (t,), t in fmt.stop))
                        if t in fmt.stop and status[m] is None:     # early accept: a box beats "none"
                            status[m] = "box"; best[m] = (nv, list(g + (t,)))
        # lower bound for undecided: best complete in heap (if any)
        k = 0
        for mm, mt in enumerate(meta):
            W, H = mt["orig_wh"]
            for q in mt["qs"]:
                m = k
                r = dict(present=bool(q["present"]), lp_none=round(L_none[m], 4), status=status[m], n_expanded=nexp[m])
                if status[m] == "box":
                    kind, box = fmt.decode(best[m][1])
                    r.update(lp_map=round(best[m][0], 4), map_box=[round(x, 5) for x in box] if box else None)
                    if q.get("box") is not None and box:
                        gt = [q["box"][0] / W, q["box"][1] / H, q["box"][2] / W, q["box"][3] / H]
                        r["iou"] = iou_np(box, gt)
                elif status[m] == "undecided":
                    comp = [(-x[0]) for x in heaps[m] if x[3]]
                    r["best_complete_lp_lower_bound"] = max(comp) if comp else None
                rows.append(r)
                k += 1
        del pc
        if bi % 25 == 0:
            st = [r["status"] for r in rows]
            print(f"[map] img {bi}/{len(dl)} q={len(rows)} fwd={n_fwd} {time.time()-t0:.0f}s none={st.count('none')} "
                  f"box={st.count('box')} undecided={st.count('undecided')}", flush=True)
    pres = [r for r in rows if r["present"]]
    absn = [r for r in rows if not r["present"]]
    f = lambda R, s: float(np.mean([r["status"] == s for r in R]))
    summ = dict(ckpt=a.ckpt, variant=cfg.variant, split_k=a.split_k, image_stride=a.image_stride, max_nodes=a.max_nodes,
                n_present=len(pres), n_absent=len(absn), wall_s=time.time() - t0, n_forward=n_fwd,
                present=dict(refuse=f(pres, "none"), accept=f(pres, "box"), undecided=f(pres, "undecided"),
                             refuse_upper=f(pres, "none") + f(pres, "undecided")),
                absent=dict(refuse=f(absn, "none"), accept=f(absn, "box"), undecided=f(absn, "undecided")),
                acc50_map_box_accepted=float(np.mean([r.get("iou", 0) >= .5 for r in pres if r["status"] == "box"])),
                mean_expanded=float(np.mean([r["n_expanded"] for r in rows])))
    print(json.dumps(summ), flush=True)
    json.dump(summ, open(a.out, "w"), indent=1)
    json.dump(rows, open(a.rows, "w"))


if __name__ == "__main__":
    main()
