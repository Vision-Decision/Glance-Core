"""Evaluate a box checkpoint.

python eval.py --ckpt /content/out/v0/ckpt_final.pt --sets refcoco_val,grefcoco_val --out results.json
  [--limit_images N] [--latency] [--set key=value ...]   (config is taken from the checkpoint, --set overrides)

Metrics
  REC sets (refcoco*/g):  acc50 = Acc@IoU>=0.5 of the predicted box (ignores presence; standard REC),
                          gated_acc50 = P(present)>=0.5 and IoU>=0.5.
  gRefCOCO:  N-acc = absent exprs with P(present)<0.5; T-acc(single) = single-target exprs with
             P(present)>=0.5 and IoU>=0.5; present_recall(multi) = multi-target exprs with P>=0.5
             (v0 has a single-box head, so multi-target boxes are not scored); AUROC of P(present).
  Calibration: ECE (15 equal-width bins) of P(present) [gRefCOCO], and of box confidence
             P(present)*P(IoU>=0.5) against the event "present and IoU>=0.5".
  Latency: batch-1 median ms per image for 1 question and for 8 questions sharing the image.
"""
import argparse, collections, json, math, os, statistics, sys, time
import numpy as np
import torch

from dv.config import Config
from dv.data.dataset import RecordDataset, collate, resize_dims
from dv.model.build import build, load_ckpt
from dv.model.heads import box_iou_giou
from train import run_model, to_dev


NUM_WORKERS = int(os.environ.get("DV_EVAL_WORKERS", "4"))  # also --num_workers

def ece(p, y, n_bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def predict(net, heads, ds, images_per_batch=8, limit_images=0):
    tok = ds.tok
    n = len(ds) if not limit_images else min(limit_images, len(ds))
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=images_per_batch,
                                     num_workers=NUM_WORKERS, collate_fn=lambda it: collate(it, tok))
    rows = []
    for batch in dl:
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        out, _ = run_model(net, heads, b, train=False)
        p = out["presence_logit"].sigmoid().cpu()
        piou = out["iou_logit"].sigmoid().cpu()
        box = out["box_pred"].clamp(0, 1).cpu()
        cmax = out["center_logits"].softmax(-1).max(-1).values.cpu()
        k = 0
        for m in meta:
            W, H = m["orig_wh"]
            for q in m["qs"]:
                bx = box[k] * torch.tensor([W, H, W, H])
                r = dict(text=q["text"], present=bool(q["present"]), p_present=float(p[k]), p_iou50=float(piou[k]),
                         center_p=float(cmax[k]), box=[round(float(v), 1) for v in bx], n_targets=q.get("n_targets", 1))
                if q.get("box") is not None:
                    iou, _ = box_iou_giou(bx[None], torch.tensor(q["box"], dtype=torch.float)[None])
                    r["iou"] = float(iou[0])
                rows.append(r)
                k += 1
    return rows


def summarize(name, rows):
    s = dict(n=len(rows))
    pres = [r for r in rows if r["present"] and "iou" in r]
    if pres:
        s["acc50"] = float(np.mean([r["iou"] >= 0.5 for r in pres]))
        s["gated_acc50"] = float(np.mean([(r["iou"] >= 0.5) and r["p_present"] >= 0.5 for r in pres]))
        s["mean_iou"] = float(np.mean([r["iou"] for r in pres]))
    absent = [r for r in rows if not r["present"]]
    if absent:
        s["N_acc"] = float(np.mean([r["p_present"] < 0.5 for r in absent]))
        s["T_acc_single"] = s.get("gated_acc50")
        multi = [r for r in rows if r["present"] and r.get("n_targets", 1) > 1]
        if multi:
            s["present_recall_multi"] = float(np.mean([r["p_present"] >= 0.5 for r in multi]))
        y = [int(r["present"]) for r in rows]
        p = [r["p_present"] for r in rows]
        s["auroc_presence"] = auroc(p, y)
        s["ece_presence"] = ece(p, y)
    # box confidence calibration on rows we can score (present single-box + absent)
    sc = [r for r in rows if ("iou" in r) or not r["present"]]
    conf = [r["p_present"] * r["p_iou50"] for r in sc]
    ev = [int(r["present"] and r.get("iou", 0) >= 0.5) for r in sc]
    s["ece_box_conf"] = ece(conf, ev)
    s["ece_iou_head_present_only"] = ece([r["p_iou50"] for r in pres], [int(r["iou"] >= 0.5) for r in pres]) if pres else None
    # kappa as a box-quality knob: AUROC of P(IoU>=0.5) against IoU>=0.5 on present single-target rows
    s["kappa_auroc"] = auroc([r["p_iou50"] for r in pres], [int(r["iou"] >= 0.5) for r in pres]) if pres else None
    return s


@torch.no_grad()
def latency(net, heads, ds, n_images=100, q_counts=(1, 8), warmup=5):
    tok = ds.tok
    res = {}
    for nq in q_counts:
        times, pre = [], []
        for i in range(n_images + warmup):
            t0 = time.perf_counter()
            it = ds[i]
            qs = (it["q_ids"] * nq)[:nq]
            it = dict(it, q_ids=qs, present=[True] * nq, box=[[0, 0, 1, 1]] * nq)
            batch = collate([it], tok)
            batch.pop("meta")
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            b = to_dev(batch, "cuda")
            out, _ = run_model(net, heads, b, train=False)
            _ = (out["presence_logit"].sigmoid().cpu(), out["box_pred"].cpu())
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            if i >= warmup:
                times.append((t3 - t2) * 1000); pre.append((t1 - t0) * 1000)
        res[f"{nq}q"] = dict(model_ms_median=statistics.median(times), model_ms_p90=float(np.percentile(times, 90)),
                             preprocess_ms_median=statistics.median(pre),
                             vis_tokens_median=None)
    return res


# ============================================================================================ v1 (typed model)
def v0_eval_to_v1(path, max_vis=640):
    recs = []
    for l in open(path):
        r = json.loads(l)
        recs.append(dict(src="eval", img=f"coco/train2014/{r['file']}", wh=[r["W"], r["H"]], max_vis=max_vis,
                         plen=0, questions=[dict(q, type="box") for q in r["questions"]]))
    return recs


def gui_to_v1(path, max_vis, upscale=1.0):
    """evals/gui_data.py output -> v1 records (absolute image paths; data_root='')."""
    recs = []
    for l in open(path):
        r = json.loads(l)
        w, h = resize_dims(*r["wh"], max_vis, upscale)
        # real prefix length: TokenBudgetSampler packs images by plen (plen=0 would pack up to 64 high-res images)
        recs.append(dict(src="gui_eval", img=os.path.join(r["image_root"], r["file"]), wh=r["wh"], max_vis=max_vis,
                         upscale=upscale, plen=(w // 32) * (h // 32) + 16, questions=[dict(q, type="box") for q in r["questions"]]))
    return recs


def zoom_pass(net, heads, recs, rows, cfg, a):
    """Crop-and-zoom second prefill: for each present-predicted question, crop a window (zoom_frac of each image side)
    centered on the first-pass box center, re-prefill it at up to zoom_max_vis tokens (upscaling allowed), map the
    second box back to full-image pixels. Presence stays from the first pass. One prefill per question."""
    flat = [(r, q) for r in recs for q in r["questions"]]
    assert len(flat) == len(rows)
    crops, idx = [], []
    for k, ((r, q), row) in enumerate(zip(flat, rows)):
        W, H = row["wh"]
        bx = row["box"]
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        cw, ch = W * a.zoom_frac, H * a.zoom_frac
        x1 = min(max(0, cx - cw / 2), W - cw); y1 = min(max(0, cy - ch / 2), H - ch)
        crop = [x1, y1, x1 + cw, y1 + ch]
        qq = dict(q, box=None)  # GT is scored after mapping back
        zw, zh = resize_dims(cw, ch, a.zoom_max_vis, 2.0)
        crops.append(dict(r, crop=crop, max_vis=a.zoom_max_vis, upscale=2.0, plen=(zw // 32) * (zh // 32) + 16,
                          questions=[qq])); idx.append(k)
    zr = predict_v1(net, heads, crops, "", cfg)
    out = []
    for k, row in enumerate(rows):
        z = zr[k]; x1, y1 = crops[k]["crop"][:2]
        bx = [z["box"][0] + x1, z["box"][1] + y1, z["box"][2] + x1, z["box"][3] + y1]
        nr = dict(row, box=[round(v, 1) for v in bx])
        g = flat[k][1].get("box")
        if g is not None:
            t = torch.tensor(bx, dtype=torch.float)[None]
            iou, _ = box_iou_giou(t, torch.tensor(g, dtype=torch.float)[None])
            nr["iou"] = float(iou[0])
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            nr["center_in_box"] = bool(g[0] <= cx <= g[2] and g[1] <= cy <= g[3])
        out.append(nr)
    return out


def venus_official(rows, eval_py):
    """Score with the official VenusBench-GD rules (inclusionAI/UI-Venus, branch VenusBench-GD, eval.py):
    positive items: normalized click point (box center) inside the normalized GT box; refusal items: correct iff the
    wrapper reports result == "negative" (ours: P(present) < 0.5). Overall = micro-average over all items."""
    import importlib.util
    fns = None
    if eval_py and os.path.exists(eval_py):
        try:
            import types
            sys.modules.setdefault("models", types.SimpleNamespace(MODEL_MAP={}))  # its model wrappers are not needed
            spec = importlib.util.spec_from_file_location("venus_eval", eval_py)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            fns = (m.eval_sample_positive_gt, m.eval_sample_negative_gt)
        except Exception as e:
            print("venus official eval.py import failed:", repr(e)[:200])
    res = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        W, H = r["wh"]; bx = r["box"]
        resp = dict(result="negative" if r["p_present"] < 0.5 else "positive",
                    point=[(bx[0] + bx[2]) / 2 / W, (bx[1] + bx[3]) / 2 / H])
        task = (r.get("meta") or {}).get("task", "?")
        if r["present"]:
            g = r["gt_box"]
            if fns:
                ok = fns[0](dict(bbox=g, img_size=[W, H]), resp) == "correct"
            else:
                gb = [g[0] / W, g[1] / H, g[2] / W, g[3] / H]; p = resp["point"]
                ok = gb[0] <= p[0] <= gb[2] and gb[1] <= p[1] <= gb[3]
        else:
            ok = (fns[1](dict(), resp) if fns else ("correct" if resp["result"] == "negative" else "wrong")) == "correct"
        res[task][0] += ok; res[task][1] += 1
        res["overall"][0] += ok; res["overall"][1] += 1
    return dict(official_eval_py=bool(fns), **{k: dict(acc=v[0] / v[1], n=v[1]) for k, v in sorted(res.items())})


def od_to_v1(path):
    """OpenDecisions items -> v1 records (one question each) + label ids per item."""
    recs = []
    for l in open(path):
        it = json.loads(l)
        st = it["state"] if isinstance(it["state"], str) else json.dumps(it["state"], ensure_ascii=False)
        q, crit = it["question"], it["question"].get("criteria")
        if q["type"] == "noul":
            crit = crit or {}
            instr = q["instructions"].strip()
            if crit:
                instr += f"\n(yes = {crit.get('true') or 'true'}; no = {crit.get('false') or 'false'})"
            qq = dict(type="noul", instr=instr, target=0.5)
        elif q["type"] == "choice":
            if isinstance(crit, dict):
                opts = [f"{k}: {v}" if v else str(k) for k, v in crit.items()]
            else:
                opts = [str(k) for k in crit]
            qq = dict(type="choice", instr=q["instructions"], options=opts, target=[1 / len(opts)] * len(opts))
        else:
            opts = [str(x) for x in crit]
            qq = dict(type="score", instr=q["instructions"], options=opts, target=[1 / len(opts)] * len(opts))
        # gold (used only for our own accuracy/ECE summary; the official numbers come from OpenDecisions' score.py)
        labs, exp = [str(x) for x in it["labels"]], str(it["expected"])
        if q["type"] == "noul":
            qq["target"] = 1.0 if exp in ("yes", "true", "1") else 0.0
        elif exp in labs:
            qq["target"] = [1.0 if l == exp else 0.0 for l in labs]
        recs.append(dict(src="opendecisions", text=st, plen=len(st) // 3 + 8, questions=[qq], od_id=it["id"],
                         od_labels=it["labels"], od_type=q["type"]))
    return recs


@torch.no_grad()
def predict_v1(net, heads, recs, data_root, cfg, prefix_budget=16384, q_budget=256):
    from dv.data.mixed import MixedDataset, collate_mixed, TokenBudgetSampler
    from train import run_model_v1
    # the token-budget sampler reorders records (length buckets); tag them so rows come back in input order, which
    # callers that pair rows with their inputs rely on (zoom_pass: the 13.2 SS-Pro zoom score was this mismatch)
    recs = [dict(r, _ri=i) for i, r in enumerate(recs)]
    ds = MixedDataset(None, data_root, net.cfg_proc, cfg, train=False, recs=recs)
    tok = ds.tok
    sampler = TokenBudgetSampler(ds, prefix_budget, q_budget, 64, shuffle=False, pool=len(ds) + 1)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=NUM_WORKERS,
                                     collate_fn=lambda it: collate_mixed(it, tok))
    rows, by_rec = [], {}
    for batch in dl:
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        res = run_model_v1(net, heads, b, train=False)
        qt = batch["qtype"]
        box_rows = {}
        multi_rows = {}
        if "box" in res:
            out = res["box"]
            bi = res["box_idx"].cpu().tolist()
            pl = out["presence_logit"].float().cpu(); piou = out["iou_logit"].sigmoid().float().cpu()
            box = out["box_pred"].clamp(0, 1).float().cpu()
            for j, m in enumerate(bi):
                box_rows[m] = (float(pl[j]), float(piou[j]), box[j])
            if "multi" in out:  # v1.2: candidates (box, heat logit, P(match|present), kappa) + count distribution
                mo = out["multi"]
                cb = mo["cand_box"].clamp(0, 1).float().cpu(); ok = mo["cand_ok"].cpu()
                mp = mo["cand_match_logit"].sigmoid().float().cpu(); ck = mo["cand_iou_logit"].sigmoid().float().cpu()
                ch = mo["cand_heat"].float().cpu(); cp = mo["count_logits"].softmax(-1).float().cpu()
                for j, m in enumerate(bi):
                    multi_rows[m] = (cb[j][ok[j]], ch[j][ok[j]], mp[j][ok[j]], ck[j][ok[j]], cp[j])
        typed_rows = {}
        if "typed_logits" in res:
            ti = res["typed_mask"].nonzero()[:, 0].cpu().tolist()
            tl = res["typed_logits"].float().cpu()
            for j, m in enumerate(ti):
                typed_rows[m] = [float(x) for x in tl[j] if x > -1e30]
        k = 0
        for mt in meta:
            W, H = mt["orig_wh"]
            rec = mt["rec"]
            rows = by_rec.setdefault(rec["_ri"], [])
            for q in mt["qs"]:
                if k in box_rows:
                    plog, pi, bx = box_rows[k]
                    bx = bx * torch.tensor([W, H, W, H])
                    r = dict(kind="box", text=q["text"], present=bool(q["present"]), p_logit=plog,
                             p_present=1 / (1 + math.exp(-plog)), p_iou50=pi, box=[round(float(v), 1) for v in bx],
                             n_targets=q.get("n_targets", 1), meta=q.get("meta"), wh=[W, H], neg=q.get("neg"))
                    r["gt_box"] = q.get("box")
                    r["gt_boxes"] = q.get("boxes")
                    if k in multi_rows:
                        cb, ch, mp, ck, cp = multi_rows[k]
                        cb = cb * torch.tensor([W, H, W, H])
                        r["cands"] = [[round(float(v), 1) for v in b] + [round(float(h), 3), round(float(p), 4), round(float(c), 4)]
                                      for b, h, p, c in zip(cb, ch, mp, ck)]
                        r["count_p"] = [round(float(x), 4) for x in cp]
                        r["boxes"], r["box_conf"] = decode_row(r, cfg.multi_match_thr, mode=cfg.multi_decode)
                    if q.get("box") is not None:
                        iou, _ = box_iou_giou(bx[None], torch.tensor(q["box"], dtype=torch.float)[None])
                        r["iou"] = float(iou[0])
                        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                        g = q["box"]
                        r["center_in_box"] = bool(g[0] <= cx <= g[2] and g[1] <= cy <= g[3])
                    rows.append(r)
                else:
                    tgt = q["target"] if q["type"] != "noul" else [1 - q["target"], q["target"]]
                    rows.append(dict(kind=q["type"], logits=typed_rows[k], target=tgt, od_id=rec.get("od_id"),
                                     od_labels=rec.get("od_labels"), src=rec.get("src"), sub=rec.get("sub")))
                k += 1
    return [x for i in sorted(by_rec) for x in by_rec[i]]


def _nll_T(logits_list, targets, T):
    tot = 0.0
    for lg, t in zip(logits_list, targets):
        x = np.asarray(lg) / T
        x = x - x.max()
        lp = x - np.log(np.exp(x).sum())
        tot -= float(np.dot(t, lp))
    return tot / max(1, len(logits_list))


def fit_temperature(logits_list, targets):
    """Minimize held-out NLL over T: coarse log grid (0.25..12, 120 points) then golden-section search in log T between
    the neighbouring grid points. (v1 used the grid only, so two types whose optima fell in the same 3.3%-wide grid cell
    got bit-identical temperatures, e.g. choice = score = 1.0461053524 in v1.)"""
    grid = np.exp(np.linspace(np.log(0.25), np.log(12), 120))
    nll = [_nll_T(logits_list, targets, T) for T in grid]
    i = int(np.argmin(nll))
    lo, hi = np.log(grid[max(i - 1, 0)]), np.log(grid[min(i + 1, len(grid) - 1)])
    f = lambda u: _nll_T(logits_list, targets, float(np.exp(u)))
    gr = (np.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(40):
        if fc < fd:
            hi, d, fd = d, c, fc; c = hi - gr * (hi - lo); fc = f(c)
        else:
            lo, c, fc = c, d, fd; d = lo + gr * (hi - lo); fd = f(d)
    T = float(np.exp((lo + hi) / 2))
    return T, dict(nll_T1=_nll_T(logits_list, targets, 1.0), nll_best=float(_nll_T(logits_list, targets, T)),
                   grid_T=float(grid[i]))


def fit_temps(rows_box, rows_typed):
    temps, info = {}, {}
    pr = [r for r in rows_box if r["kind"] == "box"]
    if pr:
        temps["presence"], info["presence"] = fit_temperature([[0.0, r["p_logit"]] for r in pr],
                                                              [[1 - r["present"], float(r["present"])] for r in pr])
    for t in ("choice", "noul", "score"):
        rr = [r for r in rows_typed if r["kind"] == t]
        if rr:
            temps[t], info[t] = fit_temperature([r["logits"] for r in rr], [r["target"] for r in rr])
            info[t]["n"] = len(rr)
    return temps, info


def apply_presence_T(rows, T):
    out = []
    for r in rows:
        if r.get("kind") == "box":
            r = dict(r, p_present=1 / (1 + math.exp(-r["p_logit"] / T)))
        out.append(r)
    return out


def typed_probs(r, T=1.0):
    x = np.asarray(r["logits"]) / T
    e = np.exp(x - x.max())
    return e / e.sum()


def summarize_typed(rows, temps=None):
    s = {}
    for t in ("choice", "noul", "score"):
        rr = [r for r in rows if r["kind"] == t]
        if not rr:
            continue
        T = (temps or {}).get(t, 1.0)
        P = [typed_probs(r, T) for r in rr]
        acc = np.mean([p.argmax() == int(np.argmax(r["target"])) for p, r in zip(P, rr)])
        conf = [p.max() for p in P]
        corr = [float(p.argmax() == int(np.argmax(r["target"]))) for p, r in zip(P, rr)]
        s[t] = dict(n=len(rr), acc=float(acc), ece_top=ece(conf, corr), T=T)
    return s


def opendecisions_score(rows, od_dir, temps=None, out_path=None):
    import subprocess, tempfile
    temps = temps or {}
    fn = out_path or tempfile.mktemp(suffix=".jsonl")
    with open(fn, "w") as f:
        for r in rows:
            p = typed_probs(r, temps.get(r["kind"], 1.0))
            if r["kind"] == "noul":
                probs = {"no": float(p[0]), "yes": float(p[1])}
            else:
                probs = {lab: float(x) for lab, x in zip(r["od_labels"], p)}
            f.write(json.dumps(dict(id=r["od_id"], probs=probs)) + "\n")
    out = subprocess.run([sys.executable, f"{od_dir}/score.py", fn, "--data", f"{od_dir}/opendecisions.jsonl", "--json"],
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout)
    except Exception:
        return dict(error=out.stdout[-2000:] + out.stderr[-2000:])


# ============================================================================================ v1.2 multi-instance metrics
def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms_list(boxes, scores, thr=0.7):
    keep = []
    for i in sorted(range(len(boxes)), key=lambda i: -scores[i]):
        if all(_iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return keep


def decode_row(r, thr=0.5, nms=0.7, mode="thr"):
    """Set answer of one box row: [] if P(present) < 0.5. mode "thr": candidates with P(match|present) >= thr (NMS 0.7);
    mode "count": the top-n candidates by P(match) after NMS, n = argmax of the count head (for the ">=4" bucket: every
    candidate with P(match) >= thr, at least 4). If nothing is selected, the single-box answer. Rows without candidates
    (pre-v1.2 models) answer the single box. -> boxes, conf (= P(present) * P(match))"""
    g = r["p_present"]
    if g < 0.5:
        return [], []
    if not r.get("cands"):
        return [r["box"]], [g * r["p_iou50"]]
    if mode == "count" and r.get("count_p"):
        c = sorted(r["cands"], key=lambda x: -x[5])
        c = [c[i] for i in nms_list([x[:4] for x in c], [x[5] for x in c], nms)]
        n = int(np.argmax(r["count_p"])) + 1
        if n >= len(r["count_p"]):
            sel = [x for x in c if x[5] >= thr]
            c = sel if len(sel) >= n else c[:n]
        else:
            c = c[:n]
        return [x[:4] for x in c], [round(g * x[5], 4) for x in c]
    c = [x for x in r["cands"] if x[5] >= thr]
    if not c:
        return [r["box"]], [g * r["p_iou50"]]
    k = nms_list([x[:4] for x in c], [x[5] for x in c], nms)
    return [c[i][:4] for i in k], [round(g * c[i][5], 4) for i in k]


def set_tp(P, G, thr=0.5):
    """one-to-one matches at IoU>=thr, greedy on IoU (highest first)"""
    pairs = sorted(((_iou(p, g), i, j) for i, p in enumerate(P) for j, g in enumerate(G)), reverse=True)
    up, ug, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in up and j not in ug:
            up.add(i); ug.add(j); tp += 1
    return tp


def summarize_multi(rows, thr=None, mode="thr"):
    """Set metrics per expression (GREC-style, IoU 0.5): F1 = 2TP/(|P|+|G|) (1 if both empty); Pr@F1=1; count accuracy
    |P| == |G|; split by number of referents (0 / 1 / >=2). thr: re-decode with this P(match) threshold (None = as
    decoded). Count head (if present): accuracy of argmax P(n=1,2,3,>=4 | present) on present rows, AUROC of
    P(n>=2 | present) for multi- vs single-target, ECE of the unconditional P(n>=2) = g * P(n>=2 | present).
    Per-box confidence g * P(match | present) is scored against "candidate is a TP" (greedy in heat order, as in training)."""
    grp = collections.defaultdict(lambda: collections.defaultdict(list))
    cnt_ok, amb_p, amb_y, uamb_p, uamb_y, conf, conf_y = [], [], [], [], [], [], []
    for r in rows:
        if r.get("kind", "box") != "box":
            continue
        G = (r.get("gt_boxes") or ([r["gt_box"]] if r.get("gt_box") else [])) if r["present"] else []
        n = len(G) if r["present"] else 0
        if r["present"] and n == 0:
            continue  # present without boxes (points-only): not scorable as a set
        P = decode_row(r, thr, mode=mode)[0] if thr is not None else (r["boxes"] if "boxes" in r else decode_row(r)[0])
        tp = set_tp(P, G)
        f1 = 1.0 if (not P and not G) else 2 * tp / (len(P) + len(G))
        key = "no_target" if n == 0 else ("single" if n == 1 else "multi")
        for k in (key, "all"):
            d = grp[k]
            d["f1"].append(f1); d["exact"].append(f1 == 1.0); d["count"].append(len(P) == n)
            d["pred_n"].append(len(P))
            if n:
                d["hit_any"].append(tp > 0); d["recall"].append(tp / n)
                d["precision"].append(tp / len(P) if P else 0.0)
        if "count_p" in r:
            cp = r["count_p"]
            if n:
                cnt_ok.append(int(np.argmax(cp)) == min(n, len(cp)) - 1)
                amb_p.append(1 - cp[0]); amb_y.append(int(n >= 2))
            uamb_p.append(r["p_present"] * (1 - cp[0])); uamb_y.append(int(n >= 2))
        if r.get("cands"):
            used = set()
            for c in r["cands"]:
                best, bj = 0.0, -1
                for j, g in enumerate(G):
                    if j not in used:
                        v = _iou(c[:4], g)
                        if v > best:
                            best, bj = v, j
                hit = best >= 0.5
                if hit:
                    used.add(bj)
                conf.append(r["p_present"] * c[5]); conf_y.append(int(hit))
    out = {}
    for k, d in grp.items():
        o = dict(n=len(d["f1"]), f1=float(np.mean(d["f1"])), pr_f1_1=float(np.mean(d["exact"])),
                 count_acc=float(np.mean(d["count"])), mean_pred_n=float(np.mean(d["pred_n"])))
        if d["hit_any"]:
            o.update(hit_any=float(np.mean(d["hit_any"])), recall=float(np.mean(d["recall"])),
                     precision=float(np.mean(d["precision"])))
        out[k] = o
    if "no_target" in out:
        out["N_acc"] = out["no_target"]["pr_f1_1"]
    if cnt_ok:
        out["count_head"] = dict(acc=float(np.mean(cnt_ok)), auroc_multi_vs_single=auroc(amb_p, amb_y),
                                 ece_uncond_multi=ece(uamb_p, uamb_y))
    if conf:
        out["box_conf"] = dict(n=len(conf), ece=ece(conf, conf_y), auroc=auroc(conf, conf_y))
    return out


def summarize_grec(rows):
    """gRefCOCO GREC-style metrics for a single-box model (GREx, arXiv 2601.05244 Sec. 3.3):
    prediction set = {decoded box} if P(present)>=0.5 else {}. A sample has F1=1 iff both sets are empty, or
    |GT|=1 and IoU>=0.5 (a single box can never reach F1=1 on a multi-target sample, which is counted as wrong).
    Pr@(F1=1,IoU>=0.5) over all samples; N-acc over no-target samples; T-acc = target samples predicted non-empty."""
    ok, nt, nt_ok, t, t_ok, multi = 0, 0, 0, 0, 0, 0
    for r in rows:
        pred = r["p_present"] >= 0.5
        n = r.get("n_targets", 1) if r["present"] else 0
        if n == 0:
            nt += 1; nt_ok += (not pred); ok += (not pred)
        else:
            t += 1; t_ok += pred
            if n == 1:
                ok += pred and r.get("iou", 0) >= 0.5
            else:
                multi += 1
    return dict(n=len(rows), pr_f1_iou50=ok / max(1, len(rows)), N_acc=nt_ok / max(1, nt), T_acc=t_ok / max(1, t),
                n_no_target=nt, n_target=t, n_multi_target=multi)


def merged_grefcoco(prep_dir, split):
    """gRefCOCO <split> (no-target + multi-target) + RefCOCO-unc <split> (single-target; same images)."""
    recs = {}
    for fn in [f"{prep_dir}/eval_grefcoco_{split}.jsonl", f"{prep_dir}/eval_refcoco_{split}.jsonl"]:
        for l in open(fn):
            r = json.loads(l)
            k = r["image_id"]
            if k in recs:
                recs[k]["questions"] += r["questions"]
            else:
                recs[k] = r
    out = []
    for r in recs.values():
        out.append(dict(src="eval", img=f"coco/train2014/{r['file']}", wh=[r["W"], r["H"]], max_vis=640, plen=0,
                        questions=[dict(q, type="box", n_targets=q.get("n_targets", 1),
                                        boxes=q.get("boxes") or ([q["box"]] if q.get("box") else [])) for q in r["questions"]]))
    return out


def dump_rows(rows, path):
    """compact per-row outputs (gzip JSONL) for reliability diagrams / bootstrap CIs"""
    import gzip
    keep = ("kind", "text", "present", "p_logit", "p_present", "p_iou50", "box", "gt_box", "gt_boxes", "iou",
            "center_in_box", "n_targets", "wh", "neg", "meta", "logits", "target", "od_id", "od_labels", "src", "sub",
            "cands", "count_p", "boxes", "box_conf")
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keep if k in r and r[k] is not None}, default=str) + "\n")


def summarize_gui(rows):
    s = dict(n=len(rows))
    pres = [r for r in rows if r["present"] and "iou" in r]
    if pres:
        s["click_acc"] = float(np.mean([r["center_in_box"] for r in pres]))
        s["acc50"] = float(np.mean([r["iou"] >= 0.5 for r in pres]))
    absent = [r for r in rows if not r["present"]]
    if absent:
        s["refusal_acc"] = float(np.mean([r["p_present"] < 0.5 for r in absent]))
        s["auroc_presence"] = auroc([r["p_present"] for r in rows], [int(r["present"]) for r in rows])
    groups = collections.defaultdict(list)
    for r in pres:
        m = r.get("meta") or {}
        groups[m.get("platform", "?") + "/" + str(m.get("ui_type") or m.get("task") or m.get("group") or "")].append(r["center_in_box"])
    s["by_group"] = {k: dict(n=len(v), click_acc=float(np.mean(v))) for k, v in sorted(groups.items())}
    return s


@torch.no_grad()
def latency_v1(net, heads, cfg, data_root, box_recs, text_recs, n=100, warmup=5):
    from dv.data.mixed import MixedDataset, collate_mixed
    from train import run_model_v1
    res = {}
    for name, recs, nq in (("box_1q", box_recs, 1), ("box_8q", box_recs, 8), ("text_1q", text_recs, 1)):
        if not recs:
            continue
        ds = MixedDataset(None, data_root, net.cfg_proc, cfg, train=False, recs=recs)
        times = []
        for i in range(min(n, len(ds)) + warmup):
            it = ds[i % len(ds)]
            k = len(it["q_ids"])
            idx = [j % k for j in range(nq)]
            it = dict(it, **{f: [it[f][j] for j in idx] for f in ("q_ids", "qtype", "labs", "tgts", "present", "has_box", "box", "points", "qs")})
            batch = collate_mixed([it], ds.tok); batch.pop("meta")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            r = run_model_v1(net, heads, to_dev(batch, "cuda"), train=False)
            _ = [v.float().cpu() for v in ([r["typed_logits"]] if "typed_logits" in r else []) + ([r["box"]["box_pred"]] if "box" in r else [])]
            torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
        res[name] = dict(ms_median=statistics.median(times), ms_mean=float(np.mean(times)),
                         ms_p80=float(np.percentile(times, 80)), ms_p90=float(np.percentile(times, 90)))
    return res


def main_v1(a, sd, cfg):
    net, heads, proc = build(cfg, "cuda")
    if not a.base:
        load_ckpt(a.ckpt, net, heads)
    net.cfg_proc = proc
    net.eval(); heads.eval()
    root = cfg.data_root
    results = dict(ckpt=a.ckpt, base=a.base, step=(sd or {}).get("extra", {}).get("step"), sets={})
    sets = a.sets.split(",")
    # ---- post-hoc temperatures on held-out training-distribution splits (never on eval sets)
    temps, tinfo = {}, {}
    if a.calib_dir and os.path.exists(f"{a.calib_dir}/calib_natural.jsonl"):
        rb = predict_v1(net, heads, [json.loads(l) for l in open(f"{a.calib_dir}/calib_natural.jsonl")], root, cfg)
        rt = predict_v1(net, heads, [json.loads(l) for l in open(f"{a.calib_dir}/calib_text.jsonl")], root, cfg) \
            if os.path.exists(f"{a.calib_dir}/calib_text.jsonl") else []
        temps, tinfo = fit_temps(rb, rt)
        if a.rows_dir:
            os.makedirs(a.rows_dir, exist_ok=True)
            dump_rows(rb, f"{a.rows_dir}/calib_natural.jsonl.gz"); dump_rows(rt, f"{a.rows_dir}/calib_text.jsonl.gz")
        if os.path.exists(f"{a.calib_dir}/calib_text_hard.jsonl"):
            rh = predict_v1(net, heads, [json.loads(l) for l in open(f"{a.calib_dir}/calib_text_hard.jsonl")], root, cfg)
            th, ih = fit_temps([], rh)
            results["temps_hard_split"], results["temps_hard_split_fit"] = th, ih
            results["calib_text_hard_heldout"] = dict(raw=summarize_typed(rh), cal=summarize_typed(rh, th))
            # choice temperature from the difficulty-matched held-out split (OpenDecisions scores choice calibration
            # on its hard tier only); noul/score keep the general split's temperatures
            results["temps_choice_hard"] = dict(temps, choice=th.get("choice", temps.get("choice", 1.0)))
        results["calib_text_heldout"] = dict(raw=summarize_typed(rt), cal=summarize_typed(rt, temps))
        results["calib_natural_heldout"] = dict(raw=summarize("calib", rb),
                                                cal=summarize("calib", apply_presence_T(rb, temps.get("presence", 1.0))))
        print("temps", temps, tinfo, flush=True)
    results["temps"], results["temps_fit"] = temps, tinfo
    for name in sets:
        t0 = time.time()
        if name == "opendecisions":
            rows = predict_v1(net, heads, od_to_v1(f"{a.od_dir}/opendecisions.jsonl"), root, cfg)
            s = dict(raw=opendecisions_score(rows, a.od_dir, out_path=a.out.replace(".json", "_od_raw.jsonl")),
                     cal=opendecisions_score(rows, a.od_dir, temps, out_path=a.out.replace(".json", "_od_cal.jsonl")),
                     typed_raw=summarize_typed(rows), typed_cal=summarize_typed(rows, temps))
            if "temps_choice_hard" in results:
                th = results["temps_choice_hard"]
                s["cal_choice_hard"] = opendecisions_score(rows, a.od_dir, th, out_path=a.out.replace(".json", "_od_calhard.jsonl"))
                s["typed_cal_choice_hard"] = summarize_typed(rows, th)
        elif name in ("ss2", "sspro", "venus"):
            fn = f"{a.gui_dir}/{name}.jsonl"
            if not os.path.exists(fn):
                print("skip", name); continue
            recs = gui_to_v1(fn, a.gui_max_vis, a.gui_upscale)
            rows = predict_v1(net, heads, recs, "", cfg)
            s = dict(raw=summarize_gui(rows), cal=summarize_gui(apply_presence_T(rows, temps.get("presence", 1.0))))
            if name == "venus":
                s["official_protocol"] = venus_official(rows, a.venus_eval_py)
            if a.zoom and name in a.zoom.split(","):
                zrows = zoom_pass(net, heads, recs, rows, cfg, a)
                s["zoom"] = summarize_gui(zrows)
                if name == "venus":
                    s["zoom_official_protocol"] = venus_official(zrows, a.venus_eval_py)
        elif name.startswith("grec_"):  # grec_val / grec_testA / grec_testB : official-style GREC metrics
            sp = name.split("_", 1)[1]
            rows = predict_v1(net, heads, merged_grefcoco(a.prep_dir, sp), root, cfg)
            s = dict(raw=summarize_grec(rows), summary=summarize(name, rows), multi=summarize_multi(rows),
                     multi_count_decode=summarize_multi(rows, thr=cfg.multi_match_thr, mode="count"))
            if "presence" in temps:
                cr = apply_presence_T(rows, temps["presence"])
                s["calibrated"] = summarize_grec(cr)
                s["multi_calibrated"] = summarize_multi(cr, thr=cfg.multi_match_thr)
        else:
            rows = predict_v1(net, heads, v0_eval_to_v1(f"{a.prep_dir}/eval_{name}.jsonl"), root, cfg)
            s = summarize(name, rows)
            if any(r.get("n_targets", 1) > 1 or r.get("gt_boxes") for r in rows):
                s["multi"] = summarize_multi(rows)
                s["multi_count_decode"] = summarize_multi(rows, thr=cfg.multi_match_thr, mode="count")
            if "presence" in temps and any(not r["present"] for r in rows):
                s["calibrated"] = summarize(name, apply_presence_T(rows, temps["presence"]))
        if isinstance(s, dict):
            s["wall_s"] = time.time() - t0
        results["sets"][name] = s
        if a.rows_dir:
            os.makedirs(a.rows_dir, exist_ok=True)
            dump_rows(rows, f"{a.rows_dir}/{name}.jsonl.gz")
            if name == "sspro" and isinstance(s, dict) and "zoom" in s:
                dump_rows(zrows, f"{a.rows_dir}/{name}_zoom.jsonl.gz")
        print(name, json.dumps(s, default=str)[:1500], flush=True)
        json.dump(results, open(a.out, "w"), indent=1, default=str)
    if a.latency:
        box_recs = v0_eval_to_v1(f"{a.prep_dir}/eval_refcoco_val.jsonl")[:120]
        text_recs = od_to_v1(f"{a.od_dir}/opendecisions.jsonl")[:120] if os.path.exists(f"{a.od_dir}/opendecisions.jsonl") else []
        results["latency"] = latency_v1(net, heads, cfg, root, box_recs, text_recs)
        results["latency"]["gpu"] = torch.cuda.get_device_name()
        print("latency", json.dumps(results["latency"]), flush=True)
    json.dump(results, open(a.out, "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--sets", default="refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,"
                                       "refcocoplus_testB,refcocog_val,refcocog_test,grefcoco_val_merged")
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--limit_images", type=int, default=0)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--save_rows", default="")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--base", action="store_true", help="v1: zero-shot base model (no ckpt; typed head = LM readout)")
    ap.add_argument("--calib_dir", default="data/prep_v1", help="v1: held-out splits for post-hoc temperatures")
    ap.add_argument("--od_dir", default="/content/data/text/opendecisions")
    ap.add_argument("--gui_dir", default="/content/gui")
    ap.add_argument("--gui_max_vis", type=int, default=2048)
    ap.add_argument("--gui_upscale", type=float, default=2.0)
    ap.add_argument("--zoom", default="", help="comma list of GUI sets to also run crop-and-zoom on (e.g. sspro)")
    ap.add_argument("--zoom_frac", type=float, default=0.4)
    ap.add_argument("--zoom_max_vis", type=int, default=4096)
    ap.add_argument("--venus_eval_py", default="/content/venus_eval/eval.py")
    ap.add_argument("--rows_dir", default="", help="v1: write per-row outputs (gzip JSONL) per eval set here")
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                    help="DataLoader workers; GUI sets at 8k visual tokens overflow /dev/shm with 8 (hung v1.1 GUI eval)")
    a = ap.parse_args()
    globals()["NUM_WORKERS"] = a.num_workers
    if a.base:
        cfg = Config().update(dict(task="v1")).update(dict(s.split("=", 1) for s in a.set), from_str=True)
        return main_v1(a, None, cfg)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update(sd["cfg"]).update(dict(s.split("=", 1) for s in a.set), from_str=True)
    if cfg.task == "v1":
        return main_v1(a, sd, cfg)
    net, heads, proc = build(cfg, "cuda")
    load_ckpt(a.ckpt, net, heads)
    net.eval(); heads.eval()
    results = dict(ckpt=a.ckpt, step=sd["extra"].get("step"), sets={})
    all_rows = {}
    for name in a.sets.split(","):
        ds = RecordDataset(f"{a.prep_dir}/eval_{name}.jsonl", cfg.image_root, proc, cfg, train=False)
        t0 = time.time()
        rows = predict(net, heads, ds, limit_images=a.limit_images)
        s = summarize(name, rows)
        s["wall_s"] = time.time() - t0
        results["sets"][name] = s
        if a.rows_dir:
            os.makedirs(a.rows_dir, exist_ok=True)
            dump_rows(rows, f"{a.rows_dir}/{name}.jsonl.gz")
            if name == "sspro" and isinstance(s, dict) and "zoom" in s:
                dump_rows(zrows, f"{a.rows_dir}/{name}_zoom.jsonl.gz")
        all_rows[name] = rows
        print(name, json.dumps(s), flush=True)
    if a.latency:
        ds = RecordDataset(f"{a.prep_dir}/eval_refcoco_val.jsonl", cfg.image_root, proc, cfg, train=False)
        results["latency"] = latency(net, heads, ds)
        results["latency"]["gpu"] = torch.cuda.get_device_name()
        print("latency", json.dumps(results["latency"]), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)
    if a.save_rows:
        json.dump(all_rows, open(a.save_rows, "w"))


if __name__ == "__main__":
    main()
