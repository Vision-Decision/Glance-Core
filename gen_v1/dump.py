"""Gen-v1 read-outs (VM, GPU): grammar-constrained greedy rows + box features for a post-hoc IoU head.

python -m gen_v1.dump --ckpt CKPT --out DIR --sets grefcoco_val_merged,arcal_gtest,arhead_rcg,refcoco_testA,...,ss2,venus \
    [--feat_sets grefcoco_val_merged,arcal_gtest,arhead_rcg]

Per question (DIR/rows_<set>.jsonl.gz), from ablation/eval_ab.py ar_decode (constrained=True, force_box=True):
  p_none       q(first answer token = "none"), full-vocabulary softmax at the last question token   (presence dial)
  greedy_none  argmax of that distribution is "none"           (unconstrained greedy decision: the fixed point)
  greedy_none_c / p_none_c / p_maxcoord_c   same restricted to grammar-valid first tokens
  box          greedy decode under the coordinate grammar with the first token forced to a digit (normalized x1y1x2y2)
  lp_box / lp_none / seqmap_none, p_maxcoord, H_first(_c)   as in ablation/eval_ab.py
  + text, present, n_targets, img, wh, gt_box(es), box_px, iou (single target with box), center_in_box (GUI click)
For --feat_sets also DIR/feats/rows_<set>.json + feats_<set>.npz in ablation/ar_iou_feats.py's format (h_q, h_ans, h_end,
tok_lp, tok_mg, L), so ablation/ar_iou_head.py's head code runs unchanged; image_id = image path.
Set definitions match scripts/sprint_abs/dump.py (Glance v1 rows) and ablation/ar_iou_prep.py (H / C splits).
"""
import argparse, gzip, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dv.config import Config  # noqa: E402
from dv.data.mixed import MixedDataset, TokenBudgetSampler  # noqa: E402
from gen_v1.common import collate_gen, build_gen, gen_load  # noqa: E402
from ablation.eval_ab import ar_decode, iou_np  # noqa: E402
from gen_v1.decode import fast_decode  # noqa: E402
from train import to_dev  # noqa: E402
from eval import v0_eval_to_v1, gui_to_v1, merged_grefcoco  # noqa: E402

GUI = ("ss2", "sspro", "venus")
CHECK = []


def grec_split(prep_dir, sp):  # = scripts/sprint_abs/dump.py grec_split
    out = []
    for l in open(f"{prep_dir}/eval_grefcoco_{sp}.jsonl"):
        r = json.loads(l)
        out.append(dict(src="eval", img=f"coco/train2014/{r['file']}", wh=[r["W"], r["H"]], max_vis=640, plen=0,
                        questions=[dict(q, type="box", n_targets=q.get("n_targets", 1) if q["present"] else 0,
                                        boxes=q.get("boxes") or ([q["box"]] if q.get("box") else [])) for q in r["questions"]]))
    return out


def load_set(name, a):
    P = a.prep_dir
    if name == "grefcoco_val_merged":
        return merged_grefcoco(P, "val"), a.data_root
    if name == "arcal_gtest":  # C: gRefCOCO testA+testB, single-target with box + no-target, minus val images
        val = {r["img"] for r in merged_grefcoco(P, "val")}
        out = []
        for r in grec_split(P, "testA") + grec_split(P, "testB"):
            if r["img"] in val:
                continue
            qs = [q for q in r["questions"] if (not q["present"]) or (q.get("n_targets") == 1 and q.get("box") is not None)]
            if qs:
                out.append(dict(r, questions=qs))
        return out, a.data_root
    if name == "arhead_rcg":  # H: RefCOCOg val+test on images outside gRefCOCO val/testA/testB
        g = {r["img"] for sp in ("val", "testA", "testB") for r in grec_split(P, sp)}
        out = [r for n in ("refcocog_val", "refcocog_test") for r in v0_eval_to_v1(f"{P}/eval_{n}.jsonl") if r["img"] not in g]
        return out, a.data_root
    if name in GUI:
        return gui_to_v1(f"{a.gui_dir}/{name}.jsonl", a.gui_max_vis, a.gui_upscale), ""
    return v0_eval_to_v1(f"{P}/eval_{name}.jsonl"), a.data_root


@torch.no_grad()
def run_set(net, fmt, ds, feats, prefix_budget, q_budget):
    tok = ds.tok
    sampler = TokenBudgetSampler(ds, prefix_budget, q_budget, 64, shuffle=False, pool=len(ds) + 1)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=8 if prefix_budget > 4096 else 3,
                                     collate_fn=lambda it: collate_gen(it, tok, with_answer=False))
    rows, F = [], dict(h_q=[], h_ans=[], h_end=[], tok_lp=[], tok_mg=[], L=[])
    checked = False
    for batch in dl:
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        res, fe = fast_decode(net, fmt, b, want_feats=feats)
        if not checked:  # incremental decoder == ablation re-run decoder on the first batch
            ref = ar_decode(net, fmt, b, tok.pad, constrained=True)
            same = np.mean([r["gen"] == q["gen"] for r, q in zip(res, ref)])
            bi = [iou_np(fmt.decode(list(r["gen"]))[1], fmt.decode(list(q["gen"]))[1]) for r, q in zip(res, ref)]
            dpn = np.abs([r["p_none"] - q["p_none"] for r, q in zip(res, ref)])
            dlp = np.abs([r["lp_box"] - q["lp_box"] for r, q in zip(res, ref)])
            CHECK.append(dict(same_tokens=float(same), box_iou_mean=float(np.mean(bi)), box_iou_lt_0_9=int(np.sum(np.array(bi) < .9)),
                              max_abs_dp_none=float(dpn.max()), median_abs_dp_none=float(np.median(dpn)),
                              median_abs_dlp_box=float(np.median(dlp)), n=len(res)))
            print("[check fast vs ar_decode]", CHECK[-1], flush=True)
            # bf16 noise floor (ar_decode vs itself after re-batching, 259 gRefCOCO items): exact tokens 22%, box IoU .98,
            # median |dp_none| 1e-4. Exact-token agreement is therefore not a usable criterion; boxes and p(none) are.
            if np.mean(bi) < 0.95 or np.median(dpn) > 0.005:
                raise SystemExit("fast_decode mismatch")
            checked = True
        if feats:
            for k in F:
                F[k].append(fe[k])
        k = 0
        for mm in meta:
            W, H = mm["orig_wh"]
            rec = mm["rec"]
            for q in mm["qs"]:
                r = res[k]
                kind, box = fmt.decode(r.pop("gen"))
                r.update(text=q["text"], present=bool(q["present"]), n_targets=q.get("n_targets", 1 if q["present"] else 0),
                         parse=kind, box=[round(v, 5) for v in box] if box else None,
                         seqmap_none=bool(r["lp_none"] > r["lp_box"]), image_id=rec["img"], img=rec["img"], wh=[W, H],
                         gt_box=q.get("box"), gt_boxes=q.get("boxes"), neg=q.get("neg"), sub=rec.get("sub"))
                if box:
                    r["box_px"] = [round(box[0] * W, 1), round(box[1] * H, 1), round(box[2] * W, 1), round(box[3] * H, 1)]
                if q.get("box") is not None:
                    gt = [q["box"][0] / W, q["box"][1] / H, q["box"][2] / W, q["box"][3] / H]
                    r["iou"] = iou_np(r["box"], gt) if box else 0.0
                    if box:
                        bx = r["box_px"]; cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                        g = q["box"]
                        r["center_in_box"] = bool(g[0] <= cx <= g[2] and g[1] <= cy <= g[3])
                rows.append(r)
                k += 1
    out = None
    if feats:
        out = {k: (torch.cat(v).numpy() if torch.is_tensor(v[0]) else np.concatenate(v)) for k, v in F.items()}
    return rows, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--feat_sets", default="grefcoco_val_merged,arcal_gtest,arhead_rcg")
    ap.add_argument("--prep_dir", default="/content/dv/data/prep")
    ap.add_argument("--data_root", default="/content/data")
    ap.add_argument("--gui_dir", default="/content/gui")
    ap.add_argument("--gui_max_vis", type=int, default=2048)
    ap.add_argument("--gui_upscale", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update(sd["cfg"])
    del sd
    net, fmt, proc = build_gen(cfg, "cuda")
    gen_load(a.ckpt, net)
    net.eval()
    os.makedirs(f"{a.out}/feats", exist_ok=True)
    fs = set(a.feat_sets.split(",")) if a.feat_sets else set()
    for name in a.sets.split(","):
        fn = f"{a.out}/rows_{name}.jsonl.gz"
        if os.path.exists(fn):
            print("exists", fn); continue
        t0 = time.time()
        recs, root = load_set(name, a)
        if a.limit:
            recs = recs[: a.limit]
        ds = MixedDataset(None, root, proc, cfg, train=False, recs=recs)
        gui = name in GUI
        rows, feats = run_set(net, fmt, ds, name in fs, 4096 if gui else 16384, 64 if gui else 256)
        if feats is not None:
            json.dump(rows, open(f"{a.out}/feats/rows_{name}.json", "w"))
            np.savez(f"{a.out}/feats/feats_{name}.npz", **feats)
        with gzip.open(fn + ".tmp", "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(fn + ".tmp", fn)
        json.dump(CHECK, open(f"{a.out}/decode_check.json", "w"))
        pres = [r for r in rows if r["present"] and "iou" in r]
        ab = [r for r in rows if not r["present"]]
        print(f"[dump] {name} n={len(rows)} {time.time()-t0:.0f}s acc50={np.mean([r['iou'] >= .5 for r in pres]) if pres else None} "
              f"nacc_greedy={np.mean([r['greedy_none'] for r in ab]) if ab else None}"
              + (f" lp_check={float(np.abs(feats['tok_lp'].sum(1) - np.array([r['lp_box'] for r in rows])).max()):.2e}" if feats else ""),
              flush=True)


if __name__ == "__main__":
    main()
