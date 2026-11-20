"""CPU check that consistency groups survive dataset sampling + collation and that the loss runs on fake head outputs.
  python scripts/v12/test_groups.py --recs /content/v12/out/catfree.jsonl"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dv.config import Config
from dv.data.mixed import MixedDataset, collate_mixed
from train import consistency_loss

ap = argparse.ArgumentParser(); ap.add_argument("--recs", required=True); ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()
from transformers import AutoProcessor
cfg = Config().update(json.load(open("configs/v1.2_4b.json")))
proc = AutoProcessor.from_pretrained(cfg.model_id)
recs = [json.loads(l) for _, l in zip(range(a.n), open(a.recs))]
ds = MixedDataset(None, cfg.data_root, proc, cfg, recs=recs)
b = collate_mixed([ds[i] for i in range(len(ds))], ds.tok)
grp, anc, pres = b["grp"], b["anchor"], b["present"]
print("questions", len(grp), "in groups", int((grp >= 0).sum()), "anchors", int(anc.sum()))
M = len(grp); N = b["vis_valid"].shape[1]
out = dict(presence_logit=torch.randn(M, requires_grad=True), center_logits=torch.randn(M, N, requires_grad=True),
           box_pred=torch.rand(M, 4, requires_grad=True))
l, p = consistency_loss(out, grp, anc, pres, cfg)
l.backward()
print("loss", float(l), {k: float(v) for k, v in p.items()})
assert p["n_cons"] > 0
print("OK")
