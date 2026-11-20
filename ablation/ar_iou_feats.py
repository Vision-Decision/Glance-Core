"""AR IoU-head experiment, step 1 (VM, GPU): frozen AR checkpoint -> grammar-constrained greedy rows (identical fields
to eval_ab.predict_ar --constrained) + features of the decoded box for a box-quality head.

python -m ablation.ar_iou_feats --ckpt CKPT --sets a,b --prep_dir data/prep --out_dir OUT

Per set: OUT/rows_<set>.json (list of row dicts) and OUT/feats_<set>.npz with, per question,
  h_q   [D] fp16  final hidden state at <|box_start|> (the position that emits the first answer token;
                  the AR analogue of the gate head's question readout)
  h_ans [D] fp16  mean final hidden state over the decoded coordinate-token inputs (positions 1..L-1 of the answer)
  h_end [D] fp16  final hidden state at the decoded <|box_end|> token (has attended to the whole box)
  tok_lp [20]     log q of each decoded answer token (incl. <|box_end|>), full-vocabulary softmax, 0-padded
  tok_mg [20]     log q(chosen) - log q(best other token) at each answer position, 0-padded
  L               number of answer tokens (incl. <|box_end|>)
The hidden states come from one extra forward of [question + decoded answer] through the same prefix cache
(prefix-LM mask: answer tokens causal, so position j sees exactly what greedy step j saw).
"""
import argparse, json, os, time
import numpy as np
import torch

from ablation.common import ABConfig, ARDataset, collate_ar, build_ar, ar_load
from ablation.eval_ab import ar_decode, _pad_blocks, iou_np
from ablation.train_ab import to_dev

LMAX = 20


@torch.no_grad()
def run_set(net, fmt, ds, ipb=8):
    tok = ds.tok
    dl = torch.utils.data.DataLoader(ds, batch_size=ipb, num_workers=8,
                                     collate_fn=lambda it: collate_ar(it, tok, with_answer=False))
    rows, F = [], dict(h_q=[], h_ans=[], h_end=[], tok_lp=[], tok_mg=[], L=[])
    for batch in dl:
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        keep = {}
        res = ar_decode(net, fmt, b, tok.pad, constrained=True, keep=keep)
        M = len(res)
        dev = b["q_ids"].device
        q_tok = [b["q_ids"][m, : int(b["q_valid"][m].sum())].tolist() for m in range(M)]
        starts = b["q_pos"][0, :, 0].tolist()
        gens = [r["gen"] for r in res]
        ids, valid, pos = _pad_blocks([q + g for q, g in zip(q_tok, gens)], starts, dev, tok.pad)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hq = net.block(keep["pc"], ids, valid, pos, b["q_img"], b["q_last"])
        for m in range(M):
            s, L = int(b["q_last"][m]), len(gens[m])
            h = hq[m, s: s + L + 1]                       # positions box_start, a_1 .. a_L
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = net.logits(h[:L]).float().log_softmax(-1)
            g = torch.tensor(gens[m], device=dev)
            ch = lp.gather(1, g[:, None])[:, 0]
            lp2 = lp.scatter(1, g[:, None], -float("inf")).max(-1).values
            tl, tm = np.zeros(LMAX, np.float32), np.zeros(LMAX, np.float32)
            tl[:L] = ch.cpu().numpy(); tm[:L] = (ch - lp2).cpu().numpy()
            F["h_q"].append(h[0].half().cpu()); F["h_end"].append(h[L].half().cpu())
            F["h_ans"].append(h[1:L].float().mean(0).half().cpu() if L > 1 else h[0].half().cpu())
            F["tok_lp"].append(tl); F["tok_mg"].append(tm); F["L"].append(L)
        k = 0
        for mm in meta:
            for q in mm["qs"]:
                r = res[k]
                kind, box = fmt.decode(r.pop("gen"))
                r.update(present=bool(q["present"]), n_targets=q.get("n_targets", 1), parse=kind,
                         box=[round(v, 5) for v in box] if box else None,
                         seqmap_none=bool(r["lp_none"] > r["lp_box"]), image_id=mm["rec"]["image_id"])
                if q.get("box") is not None:
                    W, H = mm["orig_wh"]
                    gt = [q["box"][0] / W, q["box"][1] / H, q["box"][2] / W, q["box"][3] / H]
                    r["iou"] = iou_np(r["box"], gt) if box else 0.0
                rows.append(r)
                k += 1
    out = dict(h_q=torch.stack(F["h_q"]).numpy(), h_ans=torch.stack(F["h_ans"]).numpy(),
               h_end=torch.stack(F["h_end"]).numpy(), tok_lp=np.stack(F["tok_lp"]), tok_mg=np.stack(F["tok_mg"]),
               L=np.array(F["L"]))
    return rows, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_images", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ABConfig().update({k: v for k, v in sd["cfg"].items() if k in ABConfig.__dataclass_fields__})
    net, fmt, proc = build_ar(cfg, "cuda")
    ar_load(a.ckpt, net)
    net.eval()
    for name in a.sets.split(","):
        if os.path.exists(f"{a.out_dir}/feats_{name}.npz"):
            continue
        ds = ARDataset(f"{a.prep_dir}/eval_{name}.jsonl", cfg.image_root, proc, cfg, train=False, fmt=fmt)
        if a.limit_images:
            ds.recs = ds.recs[: a.limit_images]
        t0 = time.time()
        rows, feats = run_set(net, fmt, ds)
        json.dump(rows, open(f"{a.out_dir}/rows_{name}.json", "w"))
        np.savez(f"{a.out_dir}/feats_{name}.npz", **feats)
        pres = [r for r in rows if r["present"] and "iou" in r]
        print(name, dict(n=len(rows), wall_s=round(time.time() - t0, 1),
                         acc50=float(np.mean([r["iou"] >= .5 for r in pres])) if pres else None,
                         lp_check=float(np.abs(feats["tok_lp"].sum(1) - np.array([r["lp_box"] for r in rows])).max())),
              flush=True)


if __name__ == "__main__":
    main()
