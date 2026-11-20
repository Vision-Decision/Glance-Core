"""Minimal inference for decision-vision v1.x (copied into the release folder as inference.py).

    from inference import DecisionVision
    m = DecisionVision.from_pretrained("path/to/release/v1.1")           # base weights from the Hub + adapter + heads
    out = m.decide(image="street.jpg", questions=[
        {"type": "box", "text": "the man in the red jacket"},
        {"type": "noul", "instr": "Is it raining?"},
        {"type": "score", "instr": "How many people are in the image?", "options": ["0", "1", "2", "3", "4 or more"]},
    ])
    out = m.decide(text="CRM note: ...", questions=[
        {"type": "choice", "instr": "What will the customer order?", "options": ["less", "same", "more", "not mentioned"]}])

One prefill per call: the state (image and/or text) is the shared prefix, every question is a block continuing it.
Outputs (one dict per question):
  box   : {"p_present", "box": [x1,y1,x2,y2] (pixels of the input image), "p_iou50", "confidence" = p_present * p_iou50}
          v1.2 (multi-box heads) adds "boxes": every referent, [{"box", "confidence" = p_present * P(match), "p_match",
          "p_iou50"}] ([] if p_present < 0.5), and "count": [P(n=0), P(n=1), P(n=2), P(n=3), P(n>=4)] with
          P(n=0) = 1 - p_present. "box" stays the single-box answer.
  choice/score : {"probs": [...] (per option, input order), "answer": argmax index}
  noul  : {"p_yes", "probs": [p_no, p_yes]}
Probabilities use the post-hoc temperatures stored in dv_config.json (fitted on held-out training-distribution data).
Requires the `dv` package from the repository (bundled next to this file in the release) and:
  torch, transformers>=5.9 (Qwen3.5), peft, flash-linear-attention, safetensors, pillow.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors.torch import load_file

from dv.config import Config
from dv.data.mixed import MixedDataset, collate_mixed
from dv.model.build import build
from dv.model.v1forward import run_model_v1


def _to(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


class DecisionVision:
    def __init__(self, net, heads, proc, cfg, temps, device):
        self.net, self.heads, self.proc, self.cfg, self.temps, self.device = net, heads, proc, cfg, temps, device

    @classmethod
    def from_pretrained(cls, path, device="cuda", merged=False):
        dc = json.load(open(f"{path}/dv_config.json"))
        cfg = Config().update({k: v for k, v in dc["config"].items() if k in Config.__dataclass_fields__})
        if merged:
            cfg = cfg.update(dict(model_id=f"{path}/merged-bf16", lora_r=0))
        net, heads, proc = build(cfg, device, lora=not merged)
        if not merged:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(net.hf, load_file(f"{path}/adapter/adapter_model.safetensors"))
        heads.load_state_dict(load_file(f"{path}/heads.safetensors"))
        net.eval(); heads.eval()
        return cls(net, heads, proc, cfg, dc.get("temperatures", {}), device)

    @torch.no_grad()
    def decide(self, questions, image=None, text=None, max_vis_tokens=None):
        rec = dict(src="api", questions=[])
        root = ""
        if image is not None:
            rec["img"] = os.path.abspath(image)
            rec["max_vis"] = max_vis_tokens or self.cfg.max_vis_tokens
            rec["upscale"] = 1.0 if (max_vis_tokens or 0) > 2048 else self.cfg.max_upscale
        if text is not None:
            rec["text"] = text
        for q in questions:
            q = dict(q)
            if q["type"] == "box":
                q.setdefault("present", True); q.setdefault("box", None)
            elif q["type"] == "noul":
                q.setdefault("target", 0.5)
            else:
                q.setdefault("target", [1 / len(q["options"])] * len(q["options"]))
            rec["questions"].append(q)
        ds = MixedDataset(None, root, self.proc, self.cfg, train=False, recs=[rec])
        it = ds[0]
        batch = collate_mixed([it], ds.tok)
        W, H = it["orig_wh"]
        batch.pop("meta")
        res = run_model_v1(self.net, self.heads, _to(batch, self.device), train=False)
        out, bi, ti = [], 0, 0
        for q, qt in zip(rec["questions"], batch["qtype"].tolist()):
            if qt == 0:
                b = res["box"]
                Tp = self.temps.get("presence", 1.0)
                pp = torch.sigmoid(b["presence_logit"][bi] / Tp).item()
                pi = torch.sigmoid(b["iou_logit"][bi]).item()
                bx = (b["box_pred"][bi].clamp(0, 1) * torch.tensor([W, H, W, H], device=self.device)).tolist()
                o = dict(type="box", p_present=pp, box=[round(v, 1) for v in bx], p_iou50=pi, confidence=pp * pi)
                if "multi" in b:
                    from dv.model.multibox import decode_sets
                    mo = {k: v[bi:bi + 1].float().cpu() for k, v in b["multi"].items() if torch.is_tensor(v)}
                    sc = torch.tensor([W, H, W, H])
                    mp = mo["cand_match_logit"].sigmoid()
                    cpd = mo["count_logits"].softmax(-1) if getattr(self.cfg, "multi_decode", "thr") == "count" else None
                    (bs, cf), = decode_sets(torch.tensor([pp]), torch.tensor([bx]) / sc, torch.tensor([pi]),
                                            mo["cand_box"].clamp(0, 1), mo["cand_ok"].bool(), mp, self.cfg.multi_match_thr,
                                            count_p=cpd)
                    ki = mo["cand_iou_logit"].sigmoid()[0]
                    o["boxes"] = []
                    for bb, c in zip(bs, cf):
                        j = int(((mo["cand_box"][0].clamp(0, 1) - bb).abs().sum(-1)).argmin())
                        o["boxes"].append(dict(box=[round(float(v), 1) for v in bb * sc], confidence=float(c),
                                               p_match=float(c) / pp, p_iou50=float(ki[j])))
                    cp = mo["count_logits"].softmax(-1)[0].tolist()
                    o["count"] = [1 - pp] + [pp * x for x in cp]
                out.append(o)
                bi += 1
            else:
                lg = res["typed_logits"][ti]
                lg = lg[torch.isfinite(lg)] / self.temps.get(q["type"], 1.0)
                p = lg.softmax(-1).tolist()
                o = dict(type=q["type"], probs=p, answer=int(max(range(len(p)), key=p.__getitem__)))
                if q["type"] == "noul":
                    o["p_yes"] = p[1]
                out.append(o)
                ti += 1
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--image", default="")
    ap.add_argument("--merged", action="store_true")
    a = ap.parse_args()
    m = DecisionVision.from_pretrained(a.model, merged=a.merged)
    if a.image:
        print(json.dumps(m.decide(image=a.image, questions=[
            {"type": "box", "text": "the person on the left"},
            {"type": "noul", "instr": "Is there a dog in the image?"},
            {"type": "score", "instr": "How many people are in the image?", "options": ["0", "1", "2", "3", "4 or more"]}]), indent=1))
    print(json.dumps(m.decide(text="CRM note, 2025-03-04. Dana (purchasing) said they open two new shops in May and want to "
                                   "double their weekly flour order from May.", questions=[
        {"type": "choice", "instr": "What does the note say about flour orders from May?",
         "options": ["less than now", "about the same", "more than now", "not mentioned"]},
        {"type": "noul", "instr": "Is the customer opening new shops?"}]), indent=1))
