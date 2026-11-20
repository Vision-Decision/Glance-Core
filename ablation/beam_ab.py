"""True beam search (and refined-output-space beam search) for the trained AR heads.

python -m ablation.beam_ab --ckpt CKPT --out OUT.json --rows ROWS.json [--configs b4lp0,b8lp0,b4lp1,b8lp1]
       [--split_k 2,4,10 --split_beam 8] [--limit_images N] [--check]

Eval set: gRefCOCO val merged (= RefCOCO val single-target present + gRefCOCO val no-target); multi-target
expressions are skipped (they enter neither present-refusal nor N-acc). Question order = eval_ab order with
multi-target rows removed, so rows align with outputs/ablation/<run>/rows_c.json after the same filter.

Configs: b{B}lp{L}  = HF-style beam search, B beams, length_penalty L (0: pure sequence log-prob, i.e. the MAP
approximation; 1: transformers' default). "none <|box_end|>" is an ordinary hypothesis.
--split_k: refinement test (Prop. 2): every coordinate token (atomic: all bins; hier: coarse+fine; digits: the 10
digit tokens) is split into k equivalent surface forms with q/k each (see ablation/incdec.py); beam B=split_beam, lp 0.
--check: also run incremental constrained greedy and store its tokens/log-prob (equivalence with eval_ab).
"""
import argparse, json, time
import numpy as np
import torch

from ablation.common import ABConfig, ARDataset, collate_ar, build_ar, ar_load
from ablation.eval_ab import iou_np
from ablation.train_ab import to_dev
from ablation import incdec


def coord_tokens(fmt):
    if hasattr(fmt, "digit"):
        return list(fmt.digit)
    if hasattr(fmt, "N"):
        return list(range(fmt.V0, fmt.V0 + fmt.N))
    return list(range(fmt.V0, fmt.V0 + 2 * fmt.B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", default="grefcoco_val_merged")
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--configs", default="b4lp0,b8lp0,b4lp1,b8lp1")
    ap.add_argument("--split_k", default="")
    ap.add_argument("--split_beam", type=int, default=8)
    ap.add_argument("--limit_images", type=int, default=0)
    ap.add_argument("--image_stride", type=int, default=1, help="use every n-th image (subsample)")
    ap.add_argument("--ipb", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--q_rows", type=int, default=128, help="max questions x beams per beam_search call")
    ap.add_argument("--keep_none", action="store_true", help="corrected beam (audit Q2): 'none <end>' always a finished hypothesis")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", required=True)
    a = ap.parse_args()
    import os
    mf = os.environ.get("BEAM_MEM_FRAC") or (open("/content/out/beam_mem_frac").read().strip()
                                             if os.path.exists("/content/out/beam_mem_frac") else "")
    if mf:   # cap this process' GPU memory when it shares the GPU with a training job
        torch.cuda.set_per_process_memory_fraction(float(mf))
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ABConfig().update({k: v for k, v in sd["cfg"].items() if k in ABConfig.__dataclass_fields__})
    net, fmt, proc = build_ar(cfg, "cuda")
    ar_load(a.ckpt, net)
    net.eval()
    ds = ARDataset(f"{a.prep_dir}/eval_{a.set}.jsonl", cfg.image_root, proc, cfg, train=False, fmt=fmt)
    recs = []
    for i, r in enumerate(ds.recs):
        keep = [q for q in r["questions"] if not (q["present"] and q.get("n_targets", 1) > 1)]
        if keep:
            recs.append(dict(r, questions=keep, _orig_idx=i))
    recs = recs[:: a.image_stride]
    if a.limit_images:
        recs = recs[: a.limit_images]
    ds.recs = recs
    tok = ds.tok
    dl = torch.utils.data.DataLoader(ds, batch_size=a.ipb, num_workers=6,
                                     collate_fn=lambda it: collate_ar(it, tok, with_answer=False))
    confs = []
    for c in [c for c in a.configs.split(",") if c]:
        B, L = c[1:].split("lp")
        confs.append((c, int(B), float(L), None))
    ct = coord_tokens(fmt)
    for k in [int(x) for x in a.split_k.split(",") if x]:
        confs.append((f"split{k}_b{a.split_beam}lp0", a.split_beam, 0.0, {t: k for t in ct}))
    rows, t0, timing = [], time.time(), {c[0]: 0.0 for c in confs}
    for bi, batch in enumerate(dl):
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = net.image_features(b["pixel_values"], b["grid_thw"])
            pc = net.prefill(b, feats)
            cache, lg0 = incdec.start_blocks(net, pc, b)
            p0 = lg0.softmax(-1)
            none_lp = None
            if a.keep_none:
                _, lgn = incdec.step(net, cache, torch.full((lg0.shape[0],), fmt.none, device=lg0.device))
                none_lp = (lg0.log_softmax(-1)[:, fmt.none] + lgn.log_softmax(-1)[:, fmt.box_end]).tolist()
            # was "none" inside HF's step-1 candidate set (top-2B over grammar-valid first tokens)?
            vfirst = torch.tensor(sorted(set(fmt.allowed([]))), device=lg0.device)
            rank_none = (lg0[:, vfirst] > lg0[:, fmt.none:fmt.none + 1]).sum(-1).tolist()
            res = {}
            for name, B, L, split in confs:
                ts = time.time()
                res[name] = []
                Qn = lg0.shape[0]
                qc = max(1, a.q_rows // B)          # questions per beam_search call (bounds GPU memory)
                for q0 in range(0, Qn, qc):
                    ix = torch.arange(q0, min(Qn, q0 + qc), device=lg0.device)
                    res[name] += incdec.beam_search(net, fmt, cache.select(ix), lg0[ix], B=B, length_penalty=L,
                                                    split=split, none_lp=none_lp[q0:q0 + qc] if none_lp else None)
                torch.cuda.synchronize(); timing[name] += time.time() - ts
            g = incdec.greedy(net, fmt, cache, lg0) if a.check else None
        k = 0
        for m in meta:
            W, H = m["orig_wh"]
            for q in m["qs"]:
                r = dict(present=bool(q["present"]), p_none=float(p0[k, fmt.none]), none_rank_step1=int(rank_none[k]))
                if none_lp is not None:
                    r["lp_none_end"] = round(none_lp[k], 4)
                gt = None
                if q.get("box") is not None:
                    gt = [q["box"][0] / W, q["box"][1] / H, q["box"][2] / W, q["box"][3] / H]
                for name, *_ in confs:
                    hy = res[name][k]
                    out = []
                    for h in hy:
                        kind, box = fmt.decode(h["tokens"])
                        out.append(dict(kind=kind, box=[round(v, 5) for v in box] if box else None,
                                        lp=round(h["sum_lp"], 4), sc=round(h["score"], 4)))
                    best = out[0] if out else dict(kind="bad", box=None, lp=None, sc=None)
                    r[name] = dict(kind=best["kind"], lp=best["lp"],
                                   iou=(iou_np(best["box"], gt) if (gt and best["box"]) else (0.0 if gt else None)),
                                   nbest=out)
                if g is not None:
                    gk, glp = g[k]
                    kind, box = fmt.decode(gk)
                    r["greedy_inc"] = dict(tokens=gk, lp=glp, box=box)
                rows.append(r)
                k += 1
        if bi % 20 == 0:
            print(f"[beam] batch {bi}/{len(dl)} q={len(rows)} {time.time()-t0:.0f}s "
                  + " ".join(f"{n}:{v:.0f}s" for n, v in timing.items()), flush=True)
    # summary
    pres = [r for r in rows if r["present"]]
    absn = [r for r in rows if not r["present"]]
    summ = dict(ckpt=a.ckpt, variant=cfg.variant, keep_none=a.keep_none, n_present=len(pres), n_absent=len(absn), timing_s=timing,
                image_stride=a.image_stride, configs={})
    for name, B, L, split in confs:
        summ["configs"][name] = dict(
            beams=B, length_penalty=L, split=(next(iter(split.values())) if split else 1),
            refusal_present=float(np.mean([r[name]["kind"] == "none" for r in pres])),
            N_acc=float(np.mean([r[name]["kind"] == "none" for r in absn])),
            acc50_present_accepted=float(np.mean([r[name]["iou"] >= .5 for r in pres if r[name]["kind"] == "box"])),
            gated_acc50=float(np.mean([r[name]["kind"] == "box" and r[name]["iou"] >= .5 for r in pres])),
            bad=float(np.mean([r[name]["kind"] == "bad" for r in rows])))
        print(name, json.dumps(summ["configs"][name]), flush=True)
    json.dump(summ, open(a.out, "w"), indent=1)
    json.dump(rows, open(a.rows, "w"))


if __name__ == "__main__":
    main()
