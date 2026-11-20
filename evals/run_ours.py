"""Run the v0 box model (single shared-prefix prefill) on a GUI benchmark prepared by evals/gui_data.py.

python evals/run_ours.py --ckpt ckpt.pt --bench /content/gui/sspro.jsonl --max_vis_tokens 4000 --out rows.jsonl [--latency 50]

All questions of an image share one prefill. Per question we save P(present), P(IoU>=0.5), the peak center-patch
probability, the decoded box, and two click points: the decoded-box center (primary, "box center") and the
argmax patch center. Imports from dv/ and train.py are read-only (v0 code).
"""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image

from dv.config import Config
from dv.data.dataset import Tok, collate, resize_dims
from dv.model.build import build, load_ckpt
from dv.model.heads import box_iou_giou
from train import run_model, to_dev

Image.MAX_IMAGE_PIXELS = None


class GuiDataset(torch.utils.data.Dataset):
    def __init__(self, path, proc, cfg, limit=0):
        self.recs = [json.loads(l) for l in open(path)]
        if limit:
            self.recs = self.recs[:limit]
        self.proc, self.cfg = proc, cfg
        self.tok = Tok(proc, cfg.max_q_tokens)

    def __len__(self):
        return len(self.recs)

    def load(self, r):
        img = Image.open(os.path.join(r["image_root"], r["file"])).convert("RGB")
        W, H = img.size
        w, h = resize_dims(W, H, self.cfg.max_vis_tokens, self.cfg.max_upscale)
        img = img.resize((w, h), Image.BICUBIC)
        return self.proc.image_processor(images=[img], do_resize=False, return_tensors="pt"), (W, H), (w, h)

    def __getitem__(self, i):
        r = self.recs[i]
        pv, (W, H), (w, h) = self.load(r)
        qs = r["questions"]
        return dict(pixel_values=pv["pixel_values"], grid_thw=pv["image_grid_thw"][0],
                    q_ids=[self.tok.question(q["text"]) for q in qs], present=[bool(q["present"]) for q in qs],
                    box=[[0, 0, 0, 0]] * len(qs), rec=r, qs=qs, orig_wh=(W, H), in_wh=(w, h))


def in_box(pt, b):
    return b is not None and b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


@torch.no_grad()
def predict(net, heads, ds, images_per_batch, out_f):
    tok = ds.tok
    dl = torch.utils.data.DataLoader(ds, batch_size=images_per_batch, num_workers=8,
                                     collate_fn=lambda it: (collate(it, tok), [x["in_wh"] for x in it]))
    n = 0
    t0 = time.time()
    for batch, in_whs in dl:
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        out, _ = run_model(net, heads, b, train=False)
        p = out["presence_logit"].float().sigmoid().cpu()
        piou = out["iou_logit"].float().sigmoid().cpu()
        box = out["box_pred"].float().clamp(0, 1).cpu()
        cmax = out["center_logits"].float().softmax(-1).max(-1).values.cpu()
        pidx = out["pred_idx"].cpu()
        cen = batch["centers"]
        k = 0
        for bi, (m, inwh) in enumerate(zip(meta, in_whs)):
            W, H = m["orig_wh"]
            for q in m["qs"]:
                bx = (box[k] * torch.tensor([W, H, W, H])).tolist()
                pc = cen[bi, pidx[k]].tolist()
                pt_box = [(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2]
                pt_patch = [pc[0] * W, pc[1] * H]
                r = dict(id=q["meta"]["id"], meta=q["meta"], text=q["text"], present=bool(q["present"]),
                         gt=q["box"], wh=[W, H], in_wh=list(inwh), n_vis=int(batch["vis_valid"][bi].sum()),
                         p_present=float(p[k]), p_iou50=float(piou[k]), center_p=float(cmax[k]),
                         box=[round(v, 1) for v in bx], point=[round(v, 1) for v in pt_box],
                         point_patch=[round(v, 1) for v in pt_patch])
                if q["box"] is not None:
                    iou, _ = box_iou_giou(torch.tensor(bx)[None], torch.tensor(q["box"], dtype=torch.float)[None])
                    r["iou"] = float(iou[0])
                    r["hit"] = in_box(pt_box, q["box"])
                    r["hit_patch"] = in_box(pt_patch, q["box"])
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                k += 1
                n += 1
        out_f.flush()
    print(f"[predict] {n} questions, {len(ds)} images, {time.time()-t0:.1f}s", flush=True)


@torch.no_grad()
def latency(net, heads, ds, n_images, warmup=5):
    """batch-1 latency, 1 question per image: preprocess (decode+resize+patchify) and model (prefill+heads)."""
    tok = ds.tok
    pre, mod, nv = [], [], []
    for i in range(min(n_images + warmup, len(ds))):
        t0 = time.perf_counter()
        it = ds[i]
        it = dict(it, q_ids=it["q_ids"][:1], present=[True], box=[[0, 0, 1, 1]])
        batch = collate([it], tok); batch.pop("meta")
        t1 = time.perf_counter()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        b = to_dev(batch, "cuda")
        out, _ = run_model(net, heads, b, train=False)
        _ = (out["presence_logit"].sigmoid().cpu(), out["box_pred"].cpu())
        torch.cuda.synchronize(); t3 = time.perf_counter()
        if i >= warmup:
            pre.append((t1 - t0) * 1000); mod.append((t3 - t2) * 1000); nv.append(int(batch["vis_valid"].sum()))
    return dict(n=len(mod), model_ms_median=statistics.median(mod), model_ms_p90=float(np.percentile(mod, 90)),
                model_ms_mean=float(np.mean(mod)), preprocess_ms_median=statistics.median(pre),
                vis_tokens_median=statistics.median(nv), gpu=torch.cuda.get_device_name())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--max_vis_tokens", type=int, default=640)
    ap.add_argument("--max_q_tokens", type=int, default=96)
    ap.add_argument("--images_per_batch", type=int, default=0)
    ap.add_argument("--limit_images", type=int, default=0)
    ap.add_argument("--latency", type=int, default=0, help="also time N images batch-1 (writes <out>.latency.json)")
    ap.add_argument("--latency_only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update(sd["cfg"]).update(dict(max_vis_tokens=a.max_vis_tokens, max_q_tokens=a.max_q_tokens,
                                                  grad_ckpt=False))
    net, heads, proc = build(cfg, "cuda")
    load_ckpt(a.ckpt, net, heads)
    net.eval(); heads.eval()
    ds = GuiDataset(a.bench, proc, cfg, limit=a.limit_images)
    ipb = a.images_per_batch or max(1, min(8, 5000 // a.max_vis_tokens))
    if not a.latency_only:
        with open(a.out, "w") as f:
            predict(net, heads, ds, ipb, f)
    if a.latency:
        lat = latency(net, heads, ds, a.latency)
        lat.update(max_vis_tokens=a.max_vis_tokens, bench=a.bench)
        print("latency", json.dumps(lat), flush=True)
        json.dump(lat, open(a.out + ".latency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
