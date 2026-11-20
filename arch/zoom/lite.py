"""Minimal prefix-cache engine over dv.model.backbone primitives (bf16, eager): prefill an image prefix ONCE, keep the
per-layer state (GDN recurrent state + conv tail; post-RoPE K/V of the full-attention layers), then run any number of
question blocks against it -- including blocks that carry their own image tokens (a per-question crop), with an explicit
within-block attention mask and explicit 3D (M-RoPE) positions.

The math of a plain text block is the reference math of SharedPrefixLM.forward (same GDN continuation, same masked
attention over [prefix | block]); check_equiv() in run_zoom.py verifies it on real data.
"""
import torch
import torch.nn.functional as F

from dv.model.backbone import _gdn_mix, _gdn_core, _qkv


class Lite:
    def __init__(self, net):
        self.net = net
        self.lm = net.lm
        self.types = net.layer_types
        self.dtype = torch.bfloat16

    def rope(self, x, pos):
        return self.lm.rotary_emb(x, pos)

    # ------------------------------------------------------------------ prefix (B = 1, no padding)
    @torch.no_grad()
    def prefill(self, ids, pos, feats=None, img_mask=None):
        """ids [1,L], pos [3,1,L], feats [n,D] scattered at img_mask [1,L]. Returns (cache, final-normed hidden [1,L,D])."""
        with torch.autocast("cuda", dtype=self.dtype):
            x = self.lm.embed_tokens(ids)
            if feats is not None:
                x = x.masked_scatter(img_mask[..., None], feats.to(x.dtype))
            rp = self.rope(x, pos)
            cache = []
            for layer, lt in zip(self.lm.layers, self.types):
                h = layer.input_layernorm(x)
                if lt == "linear_attention":
                    m = layer.linear_attn
                    K = m.conv_kernel_size; w = m.conv1d.weight
                    mixed, z, beta, g = _gdn_mix(m, h)
                    L = mixed.shape[-1]
                    conv = F.conv1d(mixed.to(w.dtype), w, m.conv1d.bias, padding=K - 1, groups=mixed.shape[1])[..., :L]
                    a, S = _gdn_core(m, conv, z, beta, g, None)
                    cache.append(("gdn", S, mixed[:, :, L - (K - 1):]))
                else:
                    m = layer.self_attn
                    q, k, v, gt = _qkv(m, h, *rp)
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=m.scaling, enable_gqa=True)
                    a = m.o_proj(o.transpose(1, 2).reshape(1, x.shape[1], -1) * torch.sigmoid(gt))
                    cache.append(("attn", k, v))
                x = x + a
                x = x + layer.mlp(layer.post_attention_layernorm(x))
            return cache, self.lm.norm(x)

    # ------------------------------------------------------------------ question blocks
    @torch.no_grad()
    def blocks(self, cache, ids, pos, within, feats=None, img_mask=None, prefix_keep=None, gdn_reset=False, prefix_mask_tok=None):
        """ids [M,T] (right-padded), pos [3,M,T], within [M,T,T] bool (True = may attend; must include the diagonal),
        optional feats scattered at img_mask [M,T]. All blocks continue the same (single) prefix. -> final-normed [M,T,D].
        Diagnostics: prefix_keep [Lp] bool = prefix keys the blocks may attend to in the full-attention layers (e.g. hide the
        image tokens); gdn_reset = blocks start the GDN layers from a zero state and zero conv tail (no recurrent memory)."""
        M, T = ids.shape
        with torch.autocast("cuda", dtype=self.dtype):
            x = self.lm.embed_tokens(ids)
            if feats is not None:
                x = x.masked_scatter(img_mask[..., None], feats.to(x.dtype))
            rq = self.rope(x, pos)
            mask = None
            for layer, lt, c in zip(self.lm.layers, self.types, cache):
                h = layer.input_layernorm(x)
                if lt == "linear_attention":
                    m = layer.linear_attn
                    _, S, tail = c
                    w = m.conv1d.weight
                    mixed, z, beta, g = _gdn_mix(m, h)
                    if gdn_reset:
                        tail = torch.zeros_like(tail); S = torch.zeros_like(S)
                    mq = torch.cat([tail.expand(M, -1, -1), mixed], -1).to(w.dtype)
                    conv = F.conv1d(mq, w, m.conv1d.bias, padding=0, groups=mq.shape[1])
                    a, _ = _gdn_core(m, conv, z, beta, g, S.expand(M, *S.shape[1:]).contiguous())
                else:
                    m = layer.self_attn
                    _, kp, vp = c
                    if mask is None:
                        Lp = kp.shape[2]
                        pk = torch.ones(Lp, dtype=torch.bool, device=ids.device) if prefix_keep is None else prefix_keep.to(ids.device)
                        pm = pk[None, None].expand(M, T, Lp)
                        if prefix_mask_tok is not None:   # [M,T,Lp] per-token prefix visibility
                            pm = pm & prefix_mask_tok.to(ids.device)
                        mask = torch.cat([pm, within], -1)[:, None]
                    q, k, v, gt = _qkv(m, h, *rq)
                    k = torch.cat([kp.expand(M, -1, -1, -1), k], 2); v = torch.cat([vp.expand(M, -1, -1, -1), v], 2)
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=m.scaling, enable_gqa=True)
                    a = m.o_proj(o.transpose(1, 2).reshape(M, T, -1) * torch.sigmoid(gt))
                x = x + a
                x = x + layer.mlp(layer.post_attention_layernorm(x))
            return self.lm.norm(x)


def image_positions(cur, gh, gw):
    ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    p = torch.stack([torch.full((gh * gw,), cur), cur + ii.flatten(), cur + jj.flatten()])
    return p  # [3, n]


def grid_centers(gh, gw):
    ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    return torch.stack([(jj.flatten() + 0.5) / gw, (ii.flatten() + 0.5) / gh], -1)
