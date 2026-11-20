"""Print raw generations for a few RefCOCO-val questions (parse-failure diagnosis)."""
import sys, torch
from ablation.common import ABConfig, ARDataset, collate_ar, build_ar, ar_load
from ablation.eval_ab import ar_decode
from ablation.train_ab import to_dev
ck = sys.argv[1]
sd = torch.load(ck, map_location="cpu", weights_only=False)
cfg = ABConfig().update({k: v for k, v in sd["cfg"].items() if k in ABConfig.__dataclass_fields__})
net, fmt, proc = build_ar(cfg, "cuda"); ar_load(ck, net); net.eval()
ds = ARDataset("data/prep/eval_refcoco_val.jsonl", cfg.image_root, proc, cfg, train=False, fmt=fmt)
b = collate_ar([ds[0], ds[1]], ds.tok, with_answer=False); meta = b.pop("meta")
res = ar_decode(net, fmt, to_dev(b, "cuda"), ds.tok.pad)
tgt = [a for it in [ds[0], ds[1]] for a in it["a_ids"]]
for r, t in zip(res, tgt):
    g = r["gen"]
    print("gen", [x - fmt.V0 if x >= fmt.V0 else f"t{x}" for x in g], "| tgt", [x - fmt.V0 if x >= fmt.V0 else f"t{x}" for x in t], fmt.decode(g)[0])
