"""Evaluate one ablation checkpoint and dump per-question rows.

python -m ablation.eval_ab --ckpt OUT/ckpt_final.pt --out OUT/eval.json --rows OUT/rows.json [--latency]

Per-question rows (same order for every variant -> rows can be aligned across variants):
  ours: p_present, p_iou50, box, iou, H_center (nats, over patches), H_edges (mean of 4 edge dists, nats)
  ar_*: p_none   = P(first token = "none") under the full-vocab softmax at <box_start>
        greedy_none = argmax of that distribution is "none"           (greedy / beam-1 decision)
        p_maxcoord = largest single non-"none" token prob; H_first = entropy of the first-token distribution
                     renormalised over non-"none" tokens (coordinate uncertainty, nats)
        box   = greedy decode with the first token forced to be non-"none" (standard REC Acc@0.5, ignores presence)
        lp_box / lp_none = log-prob of the decoded box sequence (incl. <box_end>) / of "none <box_end>"
        -> seqmap_none = lp_none > lp_box  (sequence-level MAP with the greedy box as candidate ~ beam search)
"""
import argparse, json, math, statistics, time
import numpy as np
import torch
import torch.nn.functional as F

from ablation.common import ABConfig, ARDataset, collate_ar, build_ar, ar_load
from ablation.dvcore.dataset import RecordDataset, collate
from ablation.dvcore.heads import box_iou_giou
from ablation.train_ab import run_ours, to_dev

SETS = "grefcoco_val_merged,refcoco_testA,refcoco_testB"


def ent(p, dim=-1):
    return -(p * p.clamp(min=1e-30).log()).sum(dim)


def iou_np(a, b):
    a = torch.tensor(a, dtype=torch.float)[None]; b = torch.tensor(b, dtype=torch.float)[None]
    return float(box_iou_giou(a, b)[0][0])


# ------------------------------------------------------------------------------------------------ ours
@torch.no_grad()
def predict_ours(net, heads, ds, ipb=8):
    tok = ds.tok
    dl = torch.utils.data.DataLoader(ds, batch_size=ipb, num_workers=8, collate_fn=lambda it: collate(it, tok))
    rows = []
    for batch in dl:
        meta = batch.pop("meta")
        out, _ = run_ours(net, heads, to_dev(batch, "cuda"), train=False)
        p = out["presence_logit"].float().sigmoid().cpu()
        piou = out["iou_logit"].float().sigmoid().cpu()
        box = out["box_pred"].clamp(0, 1).cpu()
        Hc = ent(out["center_logits"].float().softmax(-1)).cpu()
        He = ent(out["edge_logits_pred"].float().softmax(-1)).mean(-1).cpu()
        k = 0
        for m in meta:
            for q in m["qs"]:
                r = dict(present=bool(q["present"]), n_targets=q.get("n_targets", 1), p_present=float(p[k]),
                         p_iou50=float(piou[k]), H_center=float(Hc[k]), H_edges=float(He[k]),
                         box=[round(float(v), 5) for v in box[k]])
                if q.get("box") is not None:
                    W, H = m["orig_wh"]
                    gt = [q["box"][0] / W, q["box"][1] / H, q["box"][2] / W, q["box"][3] / H]
                    r["iou"] = iou_np(r["box"], gt)
                rows.append(r)
                k += 1
    return rows


# ------------------------------------------------------------------------------------------------ AR
def _pad_blocks(seqs, starts, dev, pad):
    T = max(len(s) for s in seqs)
    M = len(seqs)
    ids = torch.full((M, T), pad, dtype=torch.long)
    valid = torch.zeros(M, T, dtype=torch.bool)
    for m, s in enumerate(seqs):
        ids[m, :len(s)] = torch.tensor(s)
        valid[m, :len(s)] = True
    pos = (torch.tensor(starts)[:, None] + torch.arange(T)[None]).unsqueeze(0).expand(3, M, T).contiguous()
    return ids.to(dev), valid.to(dev), pos.to(dev)


@torch.no_grad()
def ar_decode(net, fmt, batch, pad, force_box=True, need_none_lp=True, constrained=False, keep=None):
    """batch: collate_ar(with_answer=False) on device. Returns per-question dicts.
    keep: optional dict; receives the prefix cache (keep["pc"]) for a follow-up forward (ablation/ar_iou_feats.py)."""
    dev = batch["q_ids"].device
    feats = net.image_features(batch["pixel_values"], batch["grid_thw"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pc = net.prefill(batch, feats)
    if keep is not None:
        keep["pc"] = pc
    M = batch["q_ids"].shape[0]
    q_tok = [batch["q_ids"][m, : int(batch["q_valid"][m].sum())].tolist() for m in range(M)]
    starts = batch["q_pos"][0, :, 0].tolist()
    q_img = batch["q_img"]
    q_split = batch["q_last"]
    gen = [[] for _ in range(M)]
    lp = [0.0] * M
    done = [False] * M
    res = [dict() for _ in range(M)]
    for step in range(fmt.max_len):
        act = [m for m in range(M) if not done[m]]
        if not act and not (step == 1 and need_none_lp):
            break
        seqs = [q_tok[m] + gen[m] for m in act]
        extra = []
        if step == 1 and need_none_lp:   # also score "none" -> <box_end>
            extra = [q_tok[m] + [fmt.none] for m in range(M)]
        ids, valid, pos = _pad_blocks(seqs + extra, [starts[m] for m in act] + (starts if extra else []), dev, pad)
        sel = torch.tensor(act + (list(range(M)) if extra else []), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hq = net.block(pc, ids, valid, pos, q_img[sel], q_split[sel])
            last = valid.sum(1) - 1
            lg = net.logits(hq[torch.arange(len(sel), device=dev), last]).float()
        logp = lg.log_softmax(-1)
        cmask = None
        if constrained:
            cmask = torch.zeros(len(sel), lg.shape[-1], dtype=torch.bool, device=dev)
            for i, m in enumerate(act):
                cmask[i, fmt.allowed(gen[m])] = True
            cmask = cmask[: len(act)]
        if extra:
            lpn = logp[len(act):, fmt.box_end].tolist()
            for m in range(M):
                res[m]["lp_none"] = res[m]["lp_none0"] + lpn[m]
            logp = logp[: len(act)]
        if step == 0:
            p = logp.exp()
            pn = p[:, fmt.none]
            pc_ = p.clone(); pc_[:, fmt.none] = 0
            mass = pc_.sum(-1)
            H = ent(pc_ / mass[:, None])
            top = logp.argmax(-1)
            pmax = pc_.max(-1).values
            for i, m in enumerate(act):
                res[m].update(p_none=float(pn[i]), greedy_none=bool(top[i] == fmt.none), p_maxcoord=float(pmax[i]),
                              H_first=float(H[i]), lp_none0=float(logp[i, fmt.none]))
            # grammar-valid first tokens only (constrained decoding): refuse iff P(none) > max valid coordinate token
            v = torch.zeros_like(p, dtype=torch.bool)
            v[:, fmt.allowed([])] = True
            v[:, fmt.none] = False
            pv = torch.where(v, p, torch.zeros_like(p))
            pmax_c, mass_c = pv.max(-1).values, pv.sum(-1)
            Hc = ent(pv / mass_c[:, None])
            for i, m in enumerate(act):
                res[m].update(p_maxcoord_c=float(pmax_c[i]), greedy_none_c=bool(pn[i] > pmax_c[i]),
                              p_none_c=float(pn[i] / (pn[i] + mass_c[i])), H_first_c=float(Hc[i]),
                              coord_mass_valid=float(mass_c[i]))
        choose = logp if cmask is None else logp.masked_fill(~cmask, -float("inf"))
        if step == 0 and force_box:
            choose = choose.clone(); choose[:, fmt.none] = -float("inf")
        nxt = choose.argmax(-1).tolist()
        for i, m in enumerate(act):
            t = nxt[i]
            lp[m] += float(logp[i, t])
            gen[m].append(t)
            if t in fmt.stop or (step == 0 and t == fmt.none):
                done[m] = True
    for m in range(M):
        res[m]["gen"] = gen[m]
        res[m]["lp_box"] = lp[m]
        res[m].pop("lp_none0", None)
    return res


@torch.no_grad()
def predict_ar(net, fmt, ds, ipb=8, constrained=False):
    tok = ds.tok
    dl = torch.utils.data.DataLoader(ds, batch_size=ipb, num_workers=8,
                                     collate_fn=lambda it: collate_ar(it, tok, with_answer=False))
    rows = []
    for batch in dl:
        meta = batch.pop("meta")
        res = ar_decode(net, fmt, to_dev(batch, "cuda"), tok.pad, constrained=constrained)
        k = 0
        for m in meta:
            for q in m["qs"]:
                r = res[k]
                kind, box = fmt.decode(r.pop("gen"))
                r.update(present=bool(q["present"]), n_targets=q.get("n_targets", 1), parse=kind,
                         box=[round(v, 5) for v in box] if box else None,
                         seqmap_none=bool(r["lp_none"] > r["lp_box"]))
                if q.get("box") is not None:
                    W, H = m["orig_wh"]
                    gt = [q["box"][0] / W, q["box"][1] / H, q["box"][2] / W, q["box"][3] / H]
                    r["iou"] = iou_np(r["box"], gt) if box else 0.0
                rows.append(r)
                k += 1
    return rows


# ------------------------------------------------------------------------------------------------ metrics
def ece(p, y, n_bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    from scipy.stats import rankdata
    r = rankdata(p)
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def decisions(rows, variant):
    """dict rule -> list[bool says-present]; plus presence score."""
    if variant == "ours":
        return {"gate": [r["p_present"] >= 0.5 for r in rows]}, [r["p_present"] for r in rows]
    d = {"greedy": [not r["greedy_none"] for r in rows],
         "thresh": [r["p_none"] < 0.5 for r in rows],
         "seqmap": [not r["seqmap_none"] for r in rows]}
    if rows and "greedy_none_c" in rows[0]:
        d["greedy_c"] = [not r["greedy_none_c"] for r in rows]
        d["thresh_c"] = [r["p_none_c"] < 0.5 for r in rows]
    return d, [1 - r["p_none"] for r in rows]


def summarize(rows, variant):
    s = dict(n=len(rows))
    dec, score = decisions(rows, variant)
    single = [i for i, r in enumerate(rows) if r["present"] and "iou" in r]
    absent = [i for i, r in enumerate(rows) if not r["present"]]
    if single:
        s["acc50"] = float(np.mean([rows[i]["iou"] >= 0.5 for i in single]))
        s["mean_iou"] = float(np.mean([rows[i]["iou"] for i in single]))
        if variant != "ours":
            s["parse_fail"] = float(np.mean([rows[i]["parse"] == "bad" for i in single]))
    for rule, d in dec.items():
        if single:
            s[f"gated_acc50_{rule}"] = float(np.mean([d[i] and rows[i]["iou"] >= 0.5 for i in single]))
            s[f"refusal_present_{rule}"] = float(np.mean([not d[i] for i in single]))
        if absent:
            s[f"N_acc_{rule}"] = float(np.mean([not d[i] for i in absent]))
            s[f"halluc_{rule}"] = 1 - s[f"N_acc_{rule}"]
    if absent:
        y = [int(r["present"]) for r in rows]
        s["auroc_presence"] = auroc(score, y)
        s["ece_presence"] = ece(score, y)
    return s


# ------------------------------------------------------------------------------------------------ latency
@torch.no_grad()
def latency(variant, net, heads, fmt, ds, n_images=100, q_counts=(1, 8), warmup=5):
    tok = ds.tok
    res = {}
    for nq in q_counts:
        times, steps = [], []
        for i in range(n_images + warmup):
            it = ds[i]
            qs = (it["q_ids"] * nq)[:nq]
            it = dict(it, q_ids=qs, present=[True] * nq, box=[[0, 0, 1, 1]] * nq)
            if variant == "ours":
                batch = collate([it], tok)
            else:
                it["a_ids"] = [[]] * nq
                batch = collate_ar([it], tok, with_answer=False)
            batch.pop("meta")
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            b = to_dev(batch, "cuda")
            if variant == "ours":
                out, _ = run_ours(net, heads, b, train=False)
                _ = (out["presence_logit"].sigmoid().cpu(), out["box_pred"].cpu())
                n = 0
            else:
                r = ar_decode(net, fmt, b, tok.pad, force_box=False, need_none_lp=False)
                n = max(len(x["gen"]) for x in r)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t2) * 1000); steps.append(n)
        res[f"{nq}q"] = dict(ms_median=statistics.median(times), ms_mean=float(np.mean(times)),
                             ms_p90=float(np.percentile(times, 90)), decode_steps_mean=float(np.mean(steps)))
    res["gpu"] = torch.cuda.get_device_name()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", default=SETS)
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--limit_images", type=int, default=0)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="")
    ap.add_argument("--constrained", action="store_true", help="grammar-constrained greedy decoding (AR)")
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ABConfig().update({k: v for k, v in sd["cfg"].items() if k in ABConfig.__dataclass_fields__})
    variant = cfg.variant
    heads = fmt = None
    if variant == "ours":
        from ablation.dvcore.build import build, load_ckpt
        net, heads, proc = build(cfg, "cuda")
        load_ckpt(a.ckpt, net, heads)
        heads.eval()
    else:
        net, fmt, proc = build_ar(cfg, "cuda")
        ar_load(a.ckpt, net)
    net.eval()
    results = dict(ckpt=a.ckpt, variant=variant, constrained=a.constrained, step=sd["extra"].get("step"), cfg=sd["cfg"], sets={})
    all_rows = {}

    def mk(name):
        path = f"{a.prep_dir}/eval_{name}.jsonl"
        if variant == "ours":
            ds = RecordDataset(path, cfg.image_root, proc, cfg, train=False)
        else:
            ds = ARDataset(path, cfg.image_root, proc, cfg, train=False, fmt=fmt)
        if a.limit_images:
            ds.recs = ds.recs[: a.limit_images]
        return ds

    if not a.no_eval:
        for name in a.sets.split(","):
            ds = mk(name)
            t0 = time.time()
            rows = predict_ours(net, heads, ds) if variant == "ours" else predict_ar(net, fmt, ds, constrained=a.constrained)
            s = summarize(rows, variant)
            s["wall_s"] = time.time() - t0
            if name == "grefcoco_val_merged":   # refcoco val = the single-target present rows of the merged set
                s_rc = summarize([r for r in rows if r["present"] and "iou" in r], variant)
                results["sets"]["refcoco_val"] = s_rc
            results["sets"][name] = s
            all_rows[name] = rows
            print(name, json.dumps(s), flush=True)
            json.dump(results, open(a.out, "w"), indent=1)
    if a.latency:
        results["latency"] = latency(variant, net, heads, fmt, mk("refcoco_val"))
        print("latency", json.dumps(results["latency"]), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)
    if a.rows:
        json.dump(all_rows, open(a.rows, "w"))


if __name__ == "__main__":
    main()
