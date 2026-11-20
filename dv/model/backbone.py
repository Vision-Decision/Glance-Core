"""Shared-prefix, question-block forward over a Qwen3.5-family hybrid (Gated DeltaNet + gated full attention) LLM.

Layout per step:
  prefix  : [B, Lp]  one sequence per image, LEFT-padded:  <|vision_start|> <image_pad>*n <|vision_end|>
  queries : [M, Tq]  one sequence per question block, RIGHT-padded; q_img[m] = which image it belongs to.

Every question block is an independent continuation of its image prefix (no leakage between blocks):
  * linear_attention (GDN) layers: the block starts from the prefix's final recurrent state and the prefix's
    last (k-1) conv inputs, i.e. exactly what a causal continuation would see -- but NOT the other blocks.
  * full_attention layers: the block attends to its prefix keys (all valid) + its own block
    (bidirectional within the block if cfg.attn_mask == "block_bidir", else causal).
  * prefix itself is plain causal (it never sees questions).
This is mathematically identical to running each (prefix + block_m) separately, but the prefix is computed once.

Nothing here is model-size specific: layer types / dims are read from the HF modules/config.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gdr
except Exception:  # pragma: no cover
    fla_chunk_gdr = None


def _gdr(q, k, v, g, beta, initial_state=None):
    """Chunked gated delta rule; q,k:[B,T,Hk,Dk] v:[B,T,Hv,Dv] g,beta:[B,T,Hv]. Returns (o, final_state fp32)."""
    if fla_chunk_gdr is not None:
        return fla_chunk_gdr(q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=True,
                             use_qk_l2norm_in_kernel=True)
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    return torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial_state,
                                        output_final_state=True, use_qk_l2norm_in_kernel=True)


def _gdn_mix(m, x):
    """Shared projections. x:[B,T,D] -> mixed_qkv [B,C,T], z, beta, g."""
    B, T, _ = x.shape
    mixed = m.in_proj_qkv(x).transpose(1, 2)
    z = m.in_proj_z(x).reshape(B, T, -1, m.head_v_dim)
    beta = m.in_proj_b(x).sigmoid()
    g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(x).float() + m.dt_bias)
    return mixed, z, beta, g


def _gdn_core(m, conv_out, z, beta, g, init):
    B, _, T = conv_out.shape
    conv_out = F.silu(conv_out).transpose(1, 2)
    q, k, v = torch.split(conv_out, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    q = q.reshape(B, T, -1, m.head_k_dim)
    k = k.reshape(B, T, -1, m.head_k_dim)
    v = v.reshape(B, T, -1, m.head_v_dim)
    o, S = _gdr(q, k, v, g, beta, init)
    o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(B, T, -1)
    return m.out_proj(o), S


def gdn_forward(m, xp, xq, p_valid, q_img):
    K = m.conv_kernel_size
    w = m.conv1d.weight  # [C,1,K]
    # ---- prefix (left padded; zero the pads so they leave the state untouched) ----
    xp = xp * p_valid[..., None].to(xp.dtype)
    mixed, z, beta, g = _gdn_mix(m, xp)
    Lp = mixed.shape[-1]
    conv = F.conv1d(mixed.to(w.dtype), w, m.conv1d.bias, padding=K - 1, groups=mixed.shape[1])[..., :Lp]
    op, S = _gdn_core(m, conv, z, beta, g, None)
    tail = mixed[:, :, Lp - (K - 1):]                      # last k-1 conv inputs of each prefix
    # ---- question blocks: continue from the prefix state ----
    mixed_q, zq, beta_q, g_q = _gdn_mix(m, xq)
    mq = torch.cat([tail[q_img], mixed_q], dim=-1).to(w.dtype)
    conv_q = F.conv1d(mq, w, m.conv1d.bias, padding=0, groups=mq.shape[1])  # -> exactly Tq outputs
    oq, _ = _gdn_core(m, conv_q, zq, beta_q, g_q, S[q_img].contiguous())
    return op, oq


def _qkv(m, x, cos, sin):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    B, T, _ = x.shape
    q, gate = torch.chunk(m.q_proj(x).view(B, T, -1, m.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(B, T, -1)
    q = m.q_norm(q).transpose(1, 2)
    k = m.k_norm(m.k_proj(x).view(B, T, -1, m.head_dim)).transpose(1, 2)
    v = m.v_proj(x).view(B, T, -1, m.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v, gate


def attn_forward(m, xp, xq, rope_p, rope_q, mask_p, mask_q, q_img):
    B, Lp, _ = xp.shape
    M, Tq, _ = xq.shape
    qp, kp, vp, gp = _qkv(m, xp, *rope_p)
    op = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask_p, scale=m.scaling, enable_gqa=True)
    op = op.transpose(1, 2).reshape(B, Lp, -1) * torch.sigmoid(gp)
    qq, kq, vq, gq = _qkv(m, xq, *rope_q)
    k_all = torch.cat([kp[q_img], kq], dim=2)
    v_all = torch.cat([vp[q_img], vq], dim=2)
    oq = F.scaled_dot_product_attention(qq, k_all, v_all, attn_mask=mask_q, scale=m.scaling, enable_gqa=True)
    oq = oq.transpose(1, 2).reshape(M, Tq, -1) * torch.sigmoid(gq)
    return m.o_proj(op), m.o_proj(oq)


def build_masks(p_valid, q_valid, q_img, mode):
    """Boolean SDPA masks (True = may attend)."""
    B, Lp = p_valid.shape
    M, Tq = q_valid.shape
    dev = p_valid.device
    causal = torch.ones(Lp, Lp, dtype=torch.bool, device=dev).tril()
    eye_p = torch.eye(Lp, dtype=torch.bool, device=dev)
    mask_p = (causal[None] & p_valid[:, None, :]) | eye_p[None]          # pads attend to themselves (no NaN)
    if mode == "block_bidir":
        within = q_valid[:, None, :].expand(M, Tq, Tq)
    else:
        within = torch.ones(Tq, Tq, dtype=torch.bool, device=dev).tril()[None] & q_valid[:, None, :]
    within = within | torch.eye(Tq, dtype=torch.bool, device=dev)[None]
    to_prefix = p_valid[q_img][:, None, :].expand(M, Tq, Lp)
    mask_q = torch.cat([to_prefix, within], dim=-1)
    return mask_p[:, None], mask_q[:, None]


class SharedPrefixLM(torch.nn.Module):
    """Wraps a loaded Qwen3_5ForConditionalGeneration (possibly LoRA-injected / quantized)."""

    def __init__(self, hf_model, cfg):
        super().__init__()
        self.hf = hf_model
        self.cfg = cfg
        inner = hf_model.model
        self.visual = inner.visual
        self.lm = inner.language_model
        self.layer_types = list(hf_model.config.text_config.layer_types)
        self.hidden = hf_model.config.text_config.hidden_size

    @torch.no_grad()
    def image_features(self, pixel_values, grid_thw):
        out = self.hf.model.get_image_features(pixel_values, grid_thw, return_dict=True)
        return list(out.pooler_output)

    @torch.no_grad()
    def video_features(self, pixel_values_videos, grid_thw):
        out = self.hf.model.get_video_features(pixel_values_videos, grid_thw, return_dict=True)
        return list(out.pooler_output)

    def rope(self, x, pos3):
        return self.lm.rotary_emb(x, pos3)

    def forward(self, batch, img_feats, q_feats=None):
        """batch: dict from collate (on device). img_feats: list of [n_i, D] per image.
        Optional (arch-exp in-block zoom): q_feats = list of [n, D] image features scattered into the question blocks at
        batch["q_img_mask"]; batch["within"] [M,Tq,Tq] bool overrides the within-block mask.
        Returns final-normed prefix hidden [B,Lp,D] and question hidden [M,Tq,D]."""
        emb = self.lm.embed_tokens
        xp = emb(batch["p_ids"])
        img_mask = batch["p_img_mask"]
        if img_feats:  # v1 text-only batches have no image
            xp = xp.masked_scatter(img_mask[..., None], torch.cat(img_feats, 0).to(xp.dtype))
        xq = emb(batch["q_ids"])
        if q_feats:
            xq = xq.masked_scatter(batch["q_img_mask"][..., None], torch.cat(q_feats, 0).to(xq.dtype))
        p_valid, q_valid, q_img = batch["p_valid"], batch["q_valid"], batch["q_img"]
        rope_p = self.rope(xp, batch["p_pos"])
        rope_q = self.rope(xq, batch["q_pos"])
        mask_p, mask_q = build_masks(p_valid, q_valid, q_img, self.cfg.attn_mask)
        if "within" in batch:
            Lp = p_valid.shape[1]
            to_p = mask_q[:, 0, :, :Lp]
            if "q_pmask" in batch:   # [M, Lp] prefix keys this block may attend to (in-block zoom 'alliso': hide the full image)
                to_p = to_p & batch["q_pmask"][:, None, :]
            mask_q = torch.cat([to_p, batch["within"]], -1)[:, None]

        def layer_fn(layer, ltype, xp, xq):
            hp, hq = layer.input_layernorm(xp), layer.input_layernorm(xq)
            if ltype == "linear_attention":
                ap, aq = gdn_forward(layer.linear_attn, hp, hq, p_valid, q_img)
            else:
                ap, aq = attn_forward(layer.self_attn, hp, hq, rope_p, rope_q, mask_p, mask_q, q_img)
            xp, xq = xp + ap, xq + aq
            xp = xp + layer.mlp(layer.post_attention_layernorm(xp))
            xq = xq + layer.mlp(layer.post_attention_layernorm(xq))
            return xp, xq

        for layer, ltype in zip(self.lm.layers, self.layer_types):
            if self.cfg.grad_ckpt and self.training and torch.is_grad_enabled():
                xp, xq = checkpoint(layer_fn, layer, ltype, xp, xq, use_reentrant=False)
            else:
                xp, xq = layer_fn(layer, ltype, xp, xq)
        return self.lm.norm(xp), self.lm.norm(xq)
