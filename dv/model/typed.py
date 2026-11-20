"""Typed text primitives read out at the question block's last token: choice (N options), score (K ordered levels),
noul (yes/no). All three are a softmax over the block's label set:
  choice/score: letters A.. (one per option / level);  noul: {no, yes} (each the logsumexp over "No"/"no", "Yes"/"yes").
Label logits = h @ E[label tokens]^T  (E = LM-head rows, frozen copy; Qwen3.5 ties them to the embeddings)
             + zero-init linear residual (trainable)  + per-label bias (zero-init).
At init the readout equals the base LM's next-token logits restricted to the label tokens.

Losses (per question, averaged over the typed questions of a micro-batch):
  choice: soft-target CE + w_brier * Brier          noul: soft BCE (= 2-way CE) + w_brier * Brier
  score : soft-target CE + w_score_cramer * squared Cramer (CDF L2 over levels)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

QT_CHOICE, QT_NOUL, QT_SCORE = 1, 2, 3


class TypedHead(nn.Module):
    def __init__(self, lm_head_weight, label_groups, D):
        super().__init__()
        flat = sorted({t for g in label_groups for t in g})
        self.register_buffer("E", lm_head_weight[flat].detach().float().clone())  # [V_lab, D]
        pos = {t: i for i, t in enumerate(flat)}
        G = len(label_groups)
        Gmax = max(len(g) for g in label_groups)
        gi = torch.full((G, Gmax), -1, dtype=torch.long)
        for k, g in enumerate(label_groups):
            gi[k, :len(g)] = torch.tensor([pos[t] for t in g])
        self.register_buffer("group_idx", gi)
        self.res = nn.Linear(D, G)
        nn.init.zeros_(self.res.weight); nn.init.zeros_(self.res.bias)

    def forward(self, h):
        """h: [M, D] final-normed hidden at the readout token -> [M, G] label logits (fp32)."""
        h = h.float()
        tl = h @ self.E.t()                                          # [M, V_lab]
        gi = self.group_idx
        g = tl[:, gi.clamp(min=0)].masked_fill((gi < 0)[None], float("-inf"))  # [M, G, Gmax]
        return torch.logsumexp(g, -1) + self.res(h)


def typed_logits(lab_logits, lab_idx):
    """lab_logits [M,G], lab_idx [M,L] (-1 pad) -> [M,L] logits with -inf on pads."""
    x = lab_logits.gather(1, lab_idx.clamp(min=0))
    return x.masked_fill(lab_idx < 0, float("-inf"))


def typed_loss(logits, tgt, valid, qtype, cfg, temps=None):
    """logits [M,L] (-inf pads), tgt [M,L], valid [M,L] bool, qtype [M]. Returns total, parts."""
    lp = logits.log_softmax(-1).masked_fill(~valid, 0.0)
    p = lp.exp() * valid
    ce = -(tgt * lp).sum(-1)
    brier = ((p - tgt) ** 2).sum(-1)
    cram = ((p.cumsum(-1) - tgt.cumsum(-1)) ** 2 * valid).sum(-1)
    is_s = qtype == QT_SCORE
    per = ce + torch.where(is_s, cfg.w_score_cramer * cram, cfg.w_brier * brier)
    parts = {}
    for name, code in (("choice", QT_CHOICE), ("noul", QT_NOUL), ("score", QT_SCORE)):
        m = qtype == code
        if m.any():
            parts[f"ce_{name}"] = ce[m].mean().detach()
            parts[f"acc_{name}"] = (p[m].argmax(-1) == tgt[m].argmax(-1)).float().mean().detach()
            parts[f"n_{name}"] = float(m.sum())
    return per.mean(), parts
