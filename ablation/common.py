"""Controlled output-head ablation: shared code.

Variants (identical backbone / LoRA / data order / steps / image tokens; only the output interface changes):
  ours        presence gate + soft patch center + 4 ordinal edges + IoU head (frozen copy of dv/ in ablation/dvcore)
  ar_digits   answer "x1,y1,x2,y2" in Qwen-native 0..1000 digits (1 token per digit + ',') or "none", then <|box_end|>
  ar_hier     coarse-to-fine: per coordinate 2 new tokens (coarse c in 0..B-1, fine f in 0..B-1; q = c*B+f, B*B levels)
  ar_atomic   one new token per quantized coordinate value (n_bins levels) -> 4 tokens
All AR answers end with <|box_end|>; absent -> "none" <|box_end|>.  Training = teacher-forced token-mean CE over the
full vocabulary (+ new tokens), exactly like standard SFT.  The question block is the same as ours:
<|object_ref_start|> expr <|object_ref_end|> <|box_start|>; the answer is appended after <|box_start|>.
Attention inside a block = prefix-LM: question tokens bidirectional among themselves (identical to ours'
block_bidir), answer tokens causal; question tokens never see answer tokens.
"""
import math, random, json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ablation.dvcore.config import Config
from ablation.dvcore.dataset import RecordDataset, collate
from ablation.dvcore import backbone as bb


@dataclass
class ABConfig(Config):
    variant: str = "ours"          # ours | ar_digits | ar_hier | ar_atomic
    n_bins: int = 1000             # ar_atomic
    hier_base: int = 32            # ar_hier (levels = base**2)
    lr_newtok: float = 5e-4        # new-token embedding rows (ar_hier / ar_atomic)
    data_seed: int = 0             # dataloader shuffle generator (identical order across variants)


# ------------------------------------------------------------------------------------------------ formats
NONE_TEXT = "none"


class Fmt:
    """Answer token codec. Token ids >= V0 are new tokens (trainable embedding table, tied in/out like the base)."""
    n_new = 0
    max_len = 0

    def __init__(self, tok, V0):
        t = tok.convert_tokens_to_ids
        self.V0 = V0
        self.box_end = t("<|box_end|>")
        ids = tok(NONE_TEXT, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, ids
        self.none = ids[0]
        self.digit = [tok(str(d), add_special_tokens=False)["input_ids"][0] for d in range(10)]
        self.comma = tok(",", add_special_tokens=False)["input_ids"][0]
        self.stop = {self.box_end}

    def encode(self, box):  # box: normalized [4] or None
        if box is None:
            return [self.none, self.box_end]
        return self._enc([min(1.0, max(0.0, v)) for v in box]) + [self.box_end]

    def decode(self, ids):
        """ids: generated ids (may include box_end). Returns ('none'|'box'|'bad', box_norm or None)."""
        ids = [i for i in ids if i != self.box_end]
        if ids[:1] == [self.none]:
            return "none", None
        b = self._dec(ids)
        return ("box", b) if b is not None else ("bad", None)

    def new_token_init(self):
        """For each new token: list of base token ids whose embeddings are averaged to initialise it."""
        return []

    def allowed(self, gen):
        """Grammar-valid next tokens given the generated prefix (constrained decoding). Includes "none" at step 0."""
        if gen[:1] == [self.none]:
            return [self.box_end]
        out = self._allowed(gen)
        return out + [self.none] if not gen else out

    def digits_of(self, v):
        return [self.digit[int(c)] for c in str(int(v))]


class DigitsFmt(Fmt):
    max_len = 20  # "1000,1000,1000,1000" = 19 + box_end

    def _enc(self, b):
        out = []
        for k, v in enumerate(b):
            if k:
                out.append(self.comma)
            out += self.digits_of(round(v * 1000))
        return out

    def _allowed(self, gen):
        commas = sum(1 for t in gen if t == self.comma)
        cur = 0
        for t in reversed(gen):
            if t == self.comma:
                break
            cur += 1
        out = list(self.digit) if cur < 4 else []
        if cur > 0:
            out += [self.comma] if commas < 3 else [self.box_end]
        return out

    def _dec(self, ids):
        inv = {d: str(i) for i, d in enumerate(self.digit)}
        inv[self.comma] = ","
        s = "".join(inv.get(i, "?") for i in ids)
        parts = s.split(",")
        if len(parts) != 4 or not all(p.isdigit() for p in parts):
            return None
        return [int(p) / 1000 for p in parts]


class AtomicFmt(Fmt):
    def __init__(self, tok, V0, n_bins):
        super().__init__(tok, V0)
        self.N = n_bins
        self.n_new = n_bins
        self.max_len = 5

    def _enc(self, b):
        return [self.V0 + round(v * (self.N - 1)) for v in b]

    def _allowed(self, gen):
        return list(range(self.V0, self.V0 + self.N)) if len(gen) < 4 else [self.box_end]

    def _dec(self, ids):
        if len(ids) != 4 or not all(self.V0 <= i < self.V0 + self.N for i in ids):
            return None
        return [(i - self.V0) / (self.N - 1) for i in ids]

    def new_token_init(self):
        return [self.digits_of(round(k * 1000 / (self.N - 1))) for k in range(self.N)]


class HierFmt(Fmt):
    def __init__(self, tok, V0, base):
        super().__init__(tok, V0)
        self.B = base
        self.n_new = 2 * base  # coarse tokens [V0, V0+B), fine tokens [V0+B, V0+2B)
        self.max_len = 9

    def _enc(self, b):
        L = self.B * self.B - 1
        out = []
        for v in b:
            q = round(v * L)
            out += [self.V0 + q // self.B, self.V0 + self.B + q % self.B]
        return out

    def _allowed(self, gen):
        if len(gen) >= 8:
            return [self.box_end]
        off = self.V0 + (self.B if len(gen) % 2 else 0)
        return list(range(off, off + self.B))

    def _dec(self, ids):
        if len(ids) != 8:
            return None
        L = self.B * self.B - 1
        vals = []
        for k in range(4):
            c, f = ids[2 * k] - self.V0, ids[2 * k + 1] - self.V0 - self.B
            if not (0 <= c < self.B and 0 <= f < self.B):
                return None
            vals.append((c * self.B + f) / L)
        return vals

    def new_token_init(self):
        coarse = [self.digits_of(round(c * 1000 / self.B)) for c in range(self.B)]
        fine = [self.digits_of(f) for f in range(self.B)]
        return coarse + fine


def make_fmt(cfg, tok, V0):
    if cfg.variant == "ar_digits":
        return DigitsFmt(tok, V0)
    if cfg.variant == "ar_atomic":
        return AtomicFmt(tok, V0, cfg.n_bins)
    if cfg.variant == "ar_hier":
        return HierFmt(tok, V0, cfg.hier_base)
    raise ValueError(cfg.variant)


# ------------------------------------------------------------------------------------------------ data
class ARDataset(RecordDataset):
    def __init__(self, *a, fmt=None, **kw):
        super().__init__(*a, **kw)
        self.fmt = fmt

    def __getitem__(self, i):
        it = super().__getitem__(i)  # same RNG consumption as ours (random.sample of questions)
        it["a_ids"] = [self.fmt.encode(b if p else None) for b, p in zip(it["box"], it["present"])]
        return it


def collate_ar(items, tok, with_answer=True):
    """Question block = question tokens (+ answer tokens if with_answer). q_last = index of <box_start>."""
    q_split, ans = [], []
    items2 = []
    for it in items:
        qs = []
        for q, a in zip(it["q_ids"], it["a_ids"]):
            q_split.append(len(q) - 1)
            ans.append(a)
            qs.append(q + (a if with_answer else []))
        items2.append(dict(it, q_ids=qs))
    b = collate(items2, tok)
    M, Tq = b["q_ids"].shape
    b["q_last"] = torch.tensor(q_split)
    if with_answer:
        labels = torch.full((M, Tq), -100, dtype=torch.long)
        for m, (s, a) in enumerate(zip(q_split, ans)):
            labels[m, s: s + len(a)] = torch.tensor(a)
        b["labels"] = labels
    return b


def make_loader(ds, collate_fn, cfg, shuffle=True):
    g = torch.Generator()
    g.manual_seed(cfg.data_seed)
    return torch.utils.data.DataLoader(ds, batch_size=cfg.images_per_step, shuffle=shuffle, drop_last=False,
                                       num_workers=cfg.num_workers, collate_fn=collate_fn, generator=g,
                                       persistent_workers=cfg.num_workers > 0, pin_memory=True,
                                       worker_init_fn=_winit)


def _winit(wid):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)


# ------------------------------------------------------------------------------------------------ AR model
def prefix_lm_mask(p_valid, q_valid, q_img, q_split):
    """mask_q [M,1,Tq,Lp+Tq]: question tokens (<= q_split) bidirectional, answer tokens causal."""
    M, Tq = q_valid.shape
    dev = q_valid.device
    j = torch.arange(Tq, device=dev)
    i = j[:, None]
    is_q = j[None, :] <= q_split[:, None]                         # [M,Tq] key j is a question token
    within = (is_q[:, None, :] | (j[None, None, :] <= i[None])) & q_valid[:, None, :]
    within = within | torch.eye(Tq, dtype=torch.bool, device=dev)[None]
    Lp = p_valid.shape[1]
    to_prefix = p_valid[q_img][:, None, :].expand(M, Tq, Lp)
    return torch.cat([to_prefix, within], -1)[:, None]


class ARNet(bb.SharedPrefixLM):
    def __init__(self, hf, cfg, fmt):
        super().__init__(hf, cfg)
        self.fmt = fmt
        emb = self.lm.embed_tokens.weight
        self.V0 = emb.shape[0]
        self.new_emb = None
        if fmt.n_new:
            with torch.no_grad():
                init = torch.stack([emb[ids].float().mean(0) for ids in fmt.new_token_init()])
                init = init + 0.01 * init.std() * torch.randn_like(init)
            self.new_emb = nn.Parameter(init.to(emb.device))  # fp32, tied in/out (base model is tied)

    def embed(self, ids):
        E = self.lm.embed_tokens
        if self.new_emb is None:
            return E(ids)
        is_new = ids >= self.V0
        x = E(ids.clamp(max=self.V0 - 1))
        if is_new.any():
            x = x.clone()
            x[is_new] = self.new_emb[ids[is_new] - self.V0].to(x.dtype)
        return x

    def logits(self, h):
        lg = self.hf.lm_head(h)
        if self.new_emb is not None:
            lg = torch.cat([lg, h @ self.new_emb.to(h.dtype).T], -1)
        return lg

    # ---- training forward (layer-joint prefix/block, gradient checkpointed) ----
    def forward(self, batch, img_feats):
        xp = self.embed(batch["p_ids"])
        xp = xp.masked_scatter(batch["p_img_mask"][..., None], torch.cat(img_feats, 0).to(xp.dtype))
        xq = self.embed(batch["q_ids"])
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

    # ---- inference: prefix once, then question blocks (re-run per decode step; no leakage between blocks) ----
    @torch.no_grad()
    def prefill(self, batch, img_feats):
        xp = self.embed(batch["p_ids"])
        xp = xp.masked_scatter(batch["p_img_mask"][..., None], torch.cat(img_feats, 0).to(xp.dtype))
        p_valid = batch["p_valid"]
        rope_p = self.rope(xp, batch["p_pos"])
        mask_p = bb.build_masks(p_valid, p_valid[:, :1], torch.zeros(1, dtype=torch.long, device=xp.device),
                                "causal")[0]
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
        return dict(cache=cache, p_valid=p_valid, hp=self.lm.norm(xp))

    @torch.no_grad()
    def block(self, pc, q_ids, q_valid, q_pos, q_img, q_split):
        xq = self.embed(q_ids)
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


def build_ar(cfg, device="cuda"):
    from ablation.dvcore.build import load_base
    from peft import LoraConfig, inject_adapter_in_model
    hf, proc = load_base(cfg, device)
    hf.requires_grad_(False)
    hf.config.use_cache = False
    lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                    target_modules=cfg.lora_target, bias="none")
    inject_adapter_in_model(lc, hf)
    for n, p in hf.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
            p.data = p.data.float()
    for m in hf.modules():
        if hasattr(m, "lora_A"):
            m.cast_input_dtype_enabled = False
    V0 = hf.model.language_model.embed_tokens.weight.shape[0]
    fmt = make_fmt(cfg, proc.tokenizer, V0)
    net = ARNet(hf, cfg, fmt)
    return net, fmt, proc


def ar_state(net):
    from peft import get_peft_model_state_dict
    return dict(lora=get_peft_model_state_dict(net.hf),
                new_emb=None if net.new_emb is None else net.new_emb.detach().cpu())


def ar_load(path, net):
    from peft import set_peft_model_state_dict
    sd = torch.load(path, map_location="cpu", weights_only=False)
    res = set_peft_model_state_dict(net.hf, sd["lora"])
    bad = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
    assert not bad, bad[:5]
    if sd.get("new_emb") is not None:
        net.new_emb.data.copy_(sd["new_emb"].to(net.new_emb.device))
    return sd
