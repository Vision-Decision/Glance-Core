"""Per-request rows for a Glance-X checkpoint in one direction (VM, GPU). Nothing is thresholded or fitted here.

  python scripts/glancex/dump_rows.py --src Vision-Decision/Glance-X-4B --mode image_first --impl glancex \
      --sets refcoco_val,...,grefcoco_val_merged,grefcoco_testA,grefcoco_testB,diag \
      --prep_dir data/prep --diag /content/gxdata/diagnostic.jsonl --out /content/gx/rows/released/image_first

--impl glancex  upstream package (github.com/Vision-Decision/Glance-X @ 73ee4ed, pip install -e)
--impl dv       the same model served from dv.glancex (dv kernels and heads); used for the equivalence test

One gzip JSONL per set, one row per request:
  set, image_id, qi (index in the record), text, kind, lang, src, sent_id, present, n_targets, W, H,
  gt        target boxes, xyxy pixels ([] when absent)
  p_logit   presence logit;  iou_logit  IoU-head logit (P(IoU>=0.5));  center_top1  max of the patch-centre softmax
  box       single box (argmax centre + expected edges), clamped, xyxy pixels; this is what Acc@0.5 scores
  cand      the top-20 objectness patches: [objectness, x1, y1, x2, y2] with the box decoded at that patch
            (normalized, unclamped, before NMS). Glance-X's decode_multi = keep objectness > 0.5, greedy NMS at IoU 0.5,
            clamp; scripts/glancex/score.py redoes that offline, so other thresholds need no GPU.
Batching follows glancex.evaluate: 8 images per batch for image_first, 1 for text_first / joint (--images_per_batch).
"""
import argparse, gzip, importlib, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def impl_modules(impl):
    if impl == "glancex":
        m = dict(build=importlib.import_module("glancex.build"), data=importlib.import_module("glancex.data"),
                 run=importlib.import_module("glancex.train"), cfg=importlib.import_module("glancex.config"),
                 bb=importlib.import_module("glancex.backbone"))
    else:
        m = dict(build=importlib.import_module("dv.glancex.build"), data=importlib.import_module("dv.glancex.data"),
                 run=importlib.import_module("dv.glancex.run"), cfg=importlib.import_module("dv.glancex.config"),
                 bb=importlib.import_module("dv.glancex.backbone"))
    return m


@torch.no_grad()
def candidates(heads, o, rows_centers, out, topk=20):
    h = heads.q_norm(o["r"].float())
    v = heads.v_norm(o["vis"].float())
    prob = out["obj_logits"].sigmoid()
    k = min(topk, prob.shape[1])
    sc, idx = prob.topk(k, dim=1)
    M = h.shape[0]
    vi = o["vis_index"][:, None].expand(M, k)
    v_sel, pc = v[vi, idx], rows_centers[vi, idx]
    hk = h[:, None].expand(M, k, h.shape[-1])
    el = heads.edges_logits(hk.reshape(M * k, -1), v_sel.reshape(M * k, -1), pc.reshape(M * k, 2))
    boxes = heads.expected_box(el, pc.reshape(M * k, 2)).view(M, k, 4)
    return torch.cat([sc[..., None], boxes], -1)


def r4(x):
    return [round(float(v), 4) for v in x]


@torch.no_grad()
def dump_set(M, net, heads, ds, mode, ipb, limit_images, fout, set_name, dev="cuda"):
    tok, cfg = ds.tok, net.cfg
    n = len(ds) if not limit_images else min(limit_images, len(ds))
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=ipb, num_workers=6,
                                     collate_fn=lambda it: M["data"].collate(it, tok, cfg, mode))
    n_rows = 0
    for batch in dl:
        meta = batch.pop("meta")
        b = M["run"].to_dev(batch, dev)
        out, _, (o, rv, rc) = M["run"].forward_batch(net, heads, b, train=False)
        cand = candidates(heads, o, rc, out).cpu()
        pl, il = out["presence_logit"].float().cpu(), out["iou_logit"].float().cpu()
        box = out["box_pred"].clamp(0, 1).float().cpu()
        ctop = out["center_logits"].float().softmax(-1).max(-1).values.cpu()
        qi_of = {}
        for k, mt in enumerate(meta):
            q, rec = mt["q"], mt["rec"]
            W, H = mt["orig_wh"]
            key = id(rec)
            qi = qi_of.get(key, 0); qi_of[key] = qi + 1
            nb = int(b["n_boxes"][k])
            gt = (q.get("boxes") or ([q["box"]] if q.get("box") is not None else [])) if q["present"] else []
            row = dict(set=set_name, image_id=rec["image_id"], qi=qi, text=q["text"], kind=q.get("kind", "orig"),
                       lang=q.get("lang", "en"), src=q.get("src"), sent_id=q.get("sent_id"), present=bool(b["present"][k]),
                       n_targets=nb, W=W, H=H, gt=[[round(float(v), 2) for v in g] for g in gt[:cfg.max_targets]],
                       p_logit=round(float(pl[k]), 5), iou_logit=round(float(il[k]), 5), center_top1=round(float(ctop[k]), 5),
                       box=[round(float(v), 2) for v in (box[k] * torch.tensor([W, H, W, H]))],
                       cand=[r4(c) for c in cand[k]])
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1
    return n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF repo id, exported folder or training .pt")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--impl", default="glancex", choices=["glancex", "dv"])
    ap.add_argument("--sets", required=True)
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--diag", default="")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--images_per_batch", type=int, default=0)
    ap.add_argument("--limit_images", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    M = impl_modules(a.impl)
    mode = M["bb"].canonical_mode(a.mode)
    torch.backends.cuda.enable_cudnn_sdp(False)
    os.makedirs(a.out, exist_ok=True)
    sd = M["build"].load_state(a.src)
    cfg = M["cfg"].Config().update(sd["cfg"]).update(dict(pad_img_to=0, pad_txt_to=0, pad_txt_block_to=0, pad_t2i_block_to=0,
                                                         pad_bi_block_to=0, image_root=a.image_root))
    # CUDA: bf16 base under autocast (as upstream). CPU (smoke tests): no autocast, so the base must be fp32 like LoRA.
    net, heads, proc = M["build"].build(cfg, a.device, dtype=torch.bfloat16 if a.device == "cuda" else torch.float32)
    M["build"].load_ckpt(sd, net, heads)
    net.eval(); heads.eval()
    ipb = a.images_per_batch or (8 if mode == "i2t" else 1)
    log = dict(src=a.src, impl=a.impl, mode=mode, images_per_batch=ipb, step=(sd.get("extra") or {}).get("step"),
               cfg=dict(sd["cfg"]), sets={})
    for name in [s for s in a.sets.split(",") if s]:
        path = a.diag if name == "diag" else f"{a.prep_dir}/eval_{name}.jsonl"
        fn = f"{a.out}/{name}.jsonl.gz"
        if os.path.exists(fn + ".done"):
            print("skip", name, flush=True); continue
        ds = M["data"].RecordDataset(path, cfg.image_root, proc, cfg, train=False)
        t0 = time.time()
        with gzip.open(fn, "wt") as f:
            n = dump_set(M, net, heads, ds, mode, ipb, a.limit_images, f, name, a.device)
        dt = time.time() - t0
        log["sets"][name] = dict(rows=n, wall_s=round(dt, 1))
        open(fn + ".done", "w").write(json.dumps(log["sets"][name]))
        print(name, n, f"{dt:.0f}s", flush=True)
    json.dump(log, open(f"{a.out}/run_{int(time.time())}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
