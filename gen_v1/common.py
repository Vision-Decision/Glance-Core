"""Gen-v1: Glance v1's backbone, LoRA, data mixture and schedule, with every question answered *generatively* as text.

Interface (the only change from v1):
  box    (present, box)     -> "x1,y1,x2,y2" in 0..1000 digits, then <|box_end|>      (= ablation/common.py DigitsFmt)
  box    (present, points)  -> "x,y" of the first annotated point, then <|box_end|>  (PixMo / point-only GUI items;
                               v1 supervises presence + patch choice on these and no edges)
  box    (absent)           -> "none" <|box_end|>                                     (one token "none")
  choice / score            -> the option letter, then <|im_end|>
  noul                      -> "Yes" / "No", then <|im_end|>
Typed targets in the data are distributions (soft votes / calibrated labels); the first answer token is trained with the
soft-target cross-entropy over the full vocabulary, -sum_k t_k log q(label_k) (the expected token-level CE over the
target distribution; = plain CE for one-hot targets); the closing <|im_end|> is a hard target.
Loss per micro-batch = w_box * token-mean CE over box/point answer tokens + w_typed * token-mean CE over typed answer
tokens (v1's split into a box part and a typed part, each averaged inside the micro-batch).

Inputs are v1's exactly: same records, same prompt / question block (dv/data/mixed.py ChatTok), same shared-prefix
batching (TokenBudgetSampler, same seed), same images and visual-token caps. Answer tokens are appended to each question
block. Attention inside a block is prefix-LM: question tokens bidirectional among themselves (= v1's block_bidir),
answer tokens causal, question tokens never see answer tokens (= ablation/common.py prefix_lm_mask). At inference the
prefix is computed once and each block is re-run per decoding step (no leakage between blocks).
"""
import torch
import torch.nn.functional as F

from dv.data.mixed import MixedDataset, collate_mixed
from dv.model import backbone as bb
from ablation.common import DigitsFmt, prefix_lm_mask


class GenFmt(DigitsFmt):
    def __init__(self, tok, V0, chat):
        super().__init__(tok, V0)
        self.im_end = chat.ime
        self.letter_ids = chat.letter_ids
        one = lambda s: chat.enc(s)[0]
        assert len(chat.enc("No")) == 1 and len(chat.enc("Yes")) == 1
        self.no_tok, self.yes_tok = one("No"), one("Yes")

    def encode_point(self, pt):
        x, y = [min(1.0, max(0.0, v)) for v in pt]
        return self.digits_of(round(x * 1000)) + [self.comma] + self.digits_of(round(y * 1000)) + [self.box_end]

    def typed_answer(self, qtype, labs, tgt):
        """-> (label token ids, soft target over them, teacher-forced answer ids)"""
        ids = [self.no_tok, self.yes_tok] if qtype == 2 else [self.letter_ids[i] for i in labs]
        best = max(range(len(tgt)), key=lambda i: tgt[i])
        return ids, list(tgt), [ids[best], self.im_end]


class GenDataset(MixedDataset):
    """v1's MixedDataset (identical RNG consumption) + generative answers."""

    def __init__(self, *a, fmt=None, **kw):
        super().__init__(*a, **kw)
        self.fmt = fmt

    def __getitem__(self, i):
        it = super().__getitem__(i)
        a_ids, soft = [], []
        for t, lab, tgt, pres, hb, box, pts in zip(it["qtype"], it["labs"], it["tgts"], it["present"], it["has_box"],
                                                    it["box"], it["points"]):
            if t == 0:
                if not pres:
                    a = self.fmt.encode(None)
                elif hb:
                    a = self.fmt.encode(box)
                elif pts:
                    a = self.fmt.encode_point(pts[0])
                else:
                    a = []  # present without box or points: no target (does not occur in v1 data)
                a_ids.append(a); soft.append(None)
            else:
                ids, p, a = self.fmt.typed_answer(t, lab, tgt)
                a_ids.append(a); soft.append((ids, p))
        it["a_ids"], it["soft"] = a_ids, soft
        return it


def collate_gen(items, tok, with_answer=True):
    """Question block = question tokens (+ answer tokens). q_last = index of the last question token (it predicts the
    first answer token). labels [M,Tq] hard targets (-100 = none); soft_ids / soft_p [M,K]: soft target of the first
    answer token of typed questions (at q_last)."""
    q_split, ans, soft = [], [], []
    items2 = []
    for it in items:
        qs = []
        for q, a, s in zip(it["q_ids"], it.get("a_ids", [[]] * len(it["q_ids"])), it.get("soft", [None] * len(it["q_ids"]))):
            q_split.append(len(q) - 1)
            ans.append(a); soft.append(s)
            qs.append(q + (a if with_answer else []))
        items2.append(dict(it, q_ids=qs))
    b = collate_mixed(items2, tok)
    M, Tq = b["q_ids"].shape
    b["q_last"] = torch.tensor(q_split)
    if with_answer:
        labels = torch.full((M, Tq), -100, dtype=torch.long)
        K = max([len(s[0]) for s in soft if s] + [1])
        sid = torch.zeros(M, K, dtype=torch.long)
        sp = torch.zeros(M, K)
        for m, (s, a, sf) in enumerate(zip(q_split, ans, soft)):
            if a:
                labels[m, s: s + len(a)] = torch.tensor(a)
            if sf:
                labels[m, s] = -100  # first typed token: soft target instead
                sid[m, :len(sf[0])] = torch.tensor(sf[0]); sp[m, :len(sf[1])] = torch.tensor(sf[1])
        b["labels"], b["soft_ids"], b["soft_p"] = labels, sid, sp
    return b


class GenNet(bb.SharedPrefixLM):
    """Shared-prefix LM with a prefix-LM block mask; next-token logits from the (tied) LM head."""

    def __init__(self, hf, cfg, fmt):
        super().__init__(hf, cfg)
        self.fmt = fmt

    def logits(self, h):
        return self.hf.lm_head(h)

    def _embed_prefix(self, batch, img_feats):
        xp = self.lm.embed_tokens(batch["p_ids"])
        if img_feats:
            xp = xp.masked_scatter(batch["p_img_mask"][..., None], torch.cat(img_feats, 0).to(xp.dtype))
        return xp

    def forward(self, batch, img_feats):
        xp = self._embed_prefix(batch, img_feats)
        xq = self.lm.embed_tokens(batch["q_ids"])
        p_valid, q_valid, q_img = batch["p_valid"], batch["q_valid"], batch["q_img"]
        rope_p, rope_q = self.rope(xp, batch["p_pos"]), self.rope(xq, batch["q_pos"])
        mask_p, _ = bb.build_masks(p_valid, q_valid, q_img, "causal")
        mask_q = prefix_lm_mask(p_valid, q_valid, q_img, batch["q_last"])

        def layer_fn(layer, ltype, xp, xq):
            hp, hq = layer.input_layernorm(xp), layer.input_layernorm(xq)
            if ltype == "linear_attention":
                ap, aq = bb.gdn_forward(layer.linear_attn, hp, hq, p_valid, q_img)
            else:
                ap, aq = bb.attn_forward(layer.self_attn, hp, hq, rope_p, rope_q, mask_p, mask_q, q_img)
            xp, xq = xp + ap, xq + aq
            xp = xp + layer.mlp(layer.post_attention_layernorm(xp))
            xq = xq + layer.mlp(layer.post_attention_layernorm(xq))
            return xp, xq

        from torch.utils.checkpoint import checkpoint
        for layer, ltype in zip(self.lm.layers, self.layer_types):
            if self.cfg.grad_ckpt and self.training and torch.is_grad_enabled():
                xp, xq = checkpoint(layer_fn, layer, ltype, xp, xq, use_reentrant=False)
            else:
                xp, xq = layer_fn(layer, ltype, xp, xq)
        return self.lm.norm(xp), self.lm.norm(xq)

    # ---- inference: prefix once, then question blocks (re-run per decode step) ----
    @torch.no_grad()
    def prefill(self, batch, img_feats):
        xp = self._embed_prefix(batch, img_feats)
        p_valid = batch["p_valid"]
        rope_p = self.rope(xp, batch["p_pos"])
        mask_p = bb.build_masks(p_valid, p_valid[:, :1], torch.zeros(1, dtype=torch.long, device=xp.device), "causal")[0]
        cache = []
        for layer, ltype in zip(self.lm.layers, self.layer_types):
            hp = layer.input_layernorm(xp)
            if ltype == "linear_attention":
                m = layer.linear_attn
                K = m.conv_kernel_size
                w = m.conv1d.weight
                hp = hp * p_valid[..., None].to(hp.dtype)
                mixed, z, beta, g = bb._gdn_mix(m, hp)
                Lp = mixed.shape[-1]
                conv = F.conv1d(mixed.to(w.dtype), w, m.conv1d.bias, padding=K - 1, groups=mixed.shape[1])[..., :Lp]
                ap, S = bb._gdn_core(m, conv, z, beta, g, None)
                cache.append(("gdn", S, mixed[:, :, Lp - (K - 1):]))
            else:
                m = layer.self_attn
                B, Lp, _ = hp.shape
                qp, kp, vp, gp = bb._qkv(m, hp, *rope_p)
                op = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask_p, scale=m.scaling, enable_gqa=True)
                ap = m.o_proj(op.transpose(1, 2).reshape(B, Lp, -1) * torch.sigmoid(gp))
                cache.append(("attn", kp, vp))
            xp = xp + ap
            xp = xp + layer.mlp(layer.post_attention_layernorm(xp))
        return dict(cache=cache, p_valid=p_valid)

    @torch.no_grad()
    def block(self, pc, q_ids, q_valid, q_pos, q_img, q_split):
        xq = self.lm.embed_tokens(q_ids)
        rope_q = self.rope(xq, q_pos)
        mask_q = prefix_lm_mask(pc["p_valid"], q_valid, q_img, q_split)
        for layer, ltype, c in zip(self.lm.layers, self.layer_types, pc["cache"]):
            hq = layer.input_layernorm(xq)
            if ltype == "linear_attention":
                m = layer.linear_attn
                _, S, tail = c
                w = m.conv1d.weight
                mixed_q, zq, beta_q, g_q = bb._gdn_mix(m, hq)
                mq = torch.cat([tail[q_img], mixed_q], dim=-1).to(w.dtype)
                conv_q = F.conv1d(mq, w, m.conv1d.bias, padding=0, groups=mq.shape[1])
                aq, _ = bb._gdn_core(m, conv_q, zq, beta_q, g_q, S[q_img].contiguous())
            else:
                m = layer.self_attn
                _, kp, vp = c
                M, Tq, _ = hq.shape
                qq, kq, vq, gq = bb._qkv(m, hq, *rope_q)
                k_all = torch.cat([kp[q_img], kq], 2)
                v_all = torch.cat([vp[q_img], vq], 2)
                oq = F.scaled_dot_product_attention(qq, k_all, v_all, attn_mask=mask_q, scale=m.scaling, enable_gqa=True)
                aq = m.o_proj(oq.transpose(1, 2).reshape(M, Tq, -1) * torch.sigmoid(gq))
            xq = xq + aq
            xq = xq + layer.mlp(layer.post_attention_layernorm(xq))
        return self.lm.norm(xq)


def build_gen(cfg, device="cuda"):
    """Exactly dv.model.build.build (same base load, LoRA injection and torch RNG consumption, so LoRA init and the data
    workers' seeds match v1); the v1 heads it creates are discarded."""
    from dv.model.build import build
    from dv.data.mixed import ChatTok
    net, heads, proc = build(cfg, device)
    del heads
    V0 = net.lm.embed_tokens.weight.shape[0]
    fmt = GenFmt(proc.tokenizer, V0, ChatTok(proc, cfg))
    gen = GenNet(net.hf, cfg, fmt)
    return gen, fmt, proc


def gen_state(net):
    from peft import get_peft_model_state_dict
    return dict(lora=get_peft_model_state_dict(net.hf))


def gen_load(path, net):
    from peft import set_peft_model_state_dict
    sd = torch.load(path, map_location="cpu", weights_only=False)
    res = set_peft_model_state_dict(net.hf, sd["lora"])
    bad = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
    assert not bad, bad[:5]
    return sd


def gen_loss(net, batch, cfg):
    """w_box * token-mean CE (box/point answers) + w_typed * token-mean CE (typed answers; first token soft)."""
    feats = net.image_features(batch["pixel_values"], batch["grid_thw"]) if batch["pixel_values"] is not None else []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, hq = net(batch, feats)
    lab = batch["labels"]
    M = lab.shape[0]
    ar = torch.arange(M, device=lab.device)
    is_box = batch["qtype"] == 0
    hard = lab != -100
    soft_m = (~is_box) & (batch["soft_p"].sum(-1) > 0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg_h = net.logits(hq[hard])
        lg_s = net.logits(hq[ar[soft_m], batch["q_last"][soft_m]])
    ce_h = F.cross_entropy(lg_h.float(), lab[hard], reduction="none")
    lps = lg_s.float().log_softmax(-1)
    ce_s = -(batch["soft_p"][soft_m] * lps.gather(1, batch["soft_ids"][soft_m])).sum(-1)
    tok_box = is_box[:, None].expand_as(lab)[hard]
    total, parts = 0.0, {}
    if tok_box.any():
        lb = ce_h[tok_box].mean()
        total = total + cfg.w_box * lb
        parts["box_ce"] = lb.detach(); parts["n_box"] = float(is_box.sum())
    nt = (~tok_box).sum() + soft_m.sum()
    if nt > 0:
        lt = (ce_h[~tok_box].sum() + ce_s.sum()) / nt
        total = total + cfg.w_typed * lt
        parts["typed_ce"] = lt.detach(); parts["typed_first_ce"] = ce_s.mean().detach() if soft_m.any() else lt.detach() * 0
    with torch.no_grad():
        first = lab[ar, batch["q_last"]]
        bm = is_box & (first != -100)
        if bm.any():
            lf = net.logits(hq[ar[bm], batch["q_last"][bm]].to(torch.bfloat16)).float()
            pn = lf.softmax(-1)[:, net.fmt.none]
            pres = first[bm] != net.fmt.none
            parts["pnone_pos"] = pn[pres].mean() if pres.any() else pn.sum() * 0
            parts["pnone_neg"] = pn[~pres].mean() if (~pres).any() else pn.sum() * 0
            parts["first_acc_box"] = (lf.argmax(-1) == first[bm]).float().mean()
        if soft_m.any():
            g = lps.gather(1, batch["soft_ids"][soft_m]).masked_fill(batch["soft_ids"][soft_m] == 0, -float("inf"))
            parts["acc_typed"] = (g.argmax(-1) ==
                                  batch["soft_p"][soft_m].argmax(-1)).float().mean()
        parts["ans_tokens"] = float(hard.sum() + soft_m.sum())
    return total, parts
