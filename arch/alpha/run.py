"""LoRA-delta scaling sweep (zero training): W = W0 + alpha * dW for every LoRA module, heads unchanged.

  python -m arch.alpha.run --ckpt CKPT --alphas 0,0.5,0.7,0.85,1 --what coco --out DIR     # RefCOCO val/testA/testB + gRefCOCO val merged
  python -m arch.alpha.run --ckpt CKPT --alphas 0.5,0.7 --what sspro --out DIR             # ScreenSpot-Pro native (<=16384 tok, x1)
Per alpha: DIR/a<alpha>/<set>.jsonl.gz rows (eval.py / evals/sprint/infer.py format) + DIR/a<alpha>/summary.json.
Separates two causes of the ScreenSpot-Pro drop (59.9 base generative -> 50.7 Glance): if accuracy rises as alpha < 1,
part of the drop is forgetting in the LoRA delta, not the patch-grid readout.
"""
import argparse, gzip, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from dv.config import Config
from dv.model.build import build, load_ckpt
from arch.common import set_lora_scale
import eval as E


def coco(net, heads, cfg, a, d):
    S = {}
    sets = a.sets.split(",")
    for name in sets:
        recs = E.merged_grefcoco(a.prep_dir, "val") if name == "grefcoco_val_merged" else E.v0_eval_to_v1(f"{a.prep_dir}/eval_{name}.jsonl")
        t0 = time.time()
        rows = E.predict_v1(net, heads, recs, a.data_root, cfg)
        s = E.summarize(name, rows)
        if name == "grefcoco_val_merged":
            s["grec"] = E.summarize_grec(rows)
        s["wall_s"] = time.time() - t0
        S[name] = s
        E.dump_rows(rows, f"{d}/{name}.jsonl.gz")
        print(f"[alpha] {name}", json.dumps({k: s.get(k) for k in ("acc50", "N_acc", "auroc_presence", "ece_presence")}), flush=True)
    return S


def sspro(net, heads, cfg, a, d):
    from evals.sprint import infer as I
    recs = I.load_bench(a.bench, a.max_vis, 1.0, a.limit)
    t0 = time.time()
    rows, _ = I.run(net, heads, recs, cfg, 8, 20000)
    tag = os.path.basename(a.bench).replace(".jsonl", "")
    with gzip.open(f"{d}/{tag}.rows.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hit = [r["gt"][0] <= (r["box"][0] + r["box"][2]) / 2 <= r["gt"][2] and r["gt"][1] <= (r["box"][1] + r["box"][3]) / 2 <= r["gt"][3]
           for r in rows if r["present"] and r["gt"]]
    s = dict(sspro=dict(n=len(rows), click_acc=float(np.mean(hit)), wall_s=time.time() - t0, max_vis=a.max_vis))
    print("[alpha] sspro", json.dumps(s), flush=True)
    return s


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alphas", required=True)
    ap.add_argument("--what", required=True, choices=["coco", "sspro"])
    ap.add_argument("--sets", default="grefcoco_val_merged,refcoco_testA,refcoco_testB")
    ap.add_argument("--prep_dir", default="data/prep")
    ap.add_argument("--data_root", default="/content/data")
    ap.add_argument("--bench", default="/content/gui/sspro.jsonl")
    ap.add_argument("--max_vis", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__}).update(dict(grad_ckpt=False, max_q_tokens=96))
    net, heads, proc = build(cfg, "cuda"); load_ckpt(a.ckpt, net, heads)
    net.cfg_proc = proc; net.eval(); heads.eval()
    for al in [float(x) for x in a.alphas.split(",")]:
        d = f"{a.out}/a{al:g}"
        if os.path.exists(f"{d}/summary_{a.what}.json"):
            continue
        os.makedirs(d, exist_ok=True)
        n = set_lora_scale(net, al)
        s = coco(net, heads, cfg, a, d) if a.what == "coco" else sspro(net, heads, cfg, a, d)
        s.update(alpha=al, ckpt=a.ckpt, n_lora_modules=n)
        json.dump(s, open(f"{d}/summary_{a.what}.json", "w"), indent=1)
    print("[alpha] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
