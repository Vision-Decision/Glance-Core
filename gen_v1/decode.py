"""Incremental greedy decoding for GenNet (same outputs as ablation/eval_ab.py ar_decode, ~15x less compute).

ar_decode re-runs the whole question block at every decoding step. Here the question block is run once (prefix-LM:
question tokens bidirectional among themselves), caching per layer
  GDN layers : the recurrent state after the last *valid* question token (pads get beta = 0, g = 0, i.e. leave the state
               untouched) and the last k-1 conv inputs of the valid tokens;
  attention  : keys / values of prefix + question tokens (+ a validity mask),
and every answer token is then one causal step. Mathematically identical to ar_decode; dump.py checks it against
ar_decode on the first batch of every set (same tokens, |d log-prob| small) and falls back if not.
Optionally returns the box-quality features of ablation/ar_iou_feats.py (h_q, h_ans, h_end, tok_lp, tok_mg, L).
"""
import numpy as np
import torch
import torch.nn.functional as F

from dv.model import backbone as bb
from ablation.eval_ab import ent

LMAX = 20


def _gather_tail(mixed_all, lens, K):
    """mixed_all [M,C,(K-1)+Tq] = prefix tail ++ block inputs; -> the K-1 inputs ending at each row's last valid token."""
    M = mixed_all.shape[0]
    idx = (lens[:, None] + torch.arange(K - 1, device=lens.device)[None])  # positions (K-1)+len-(K-1) .. (K-1)+len-1
    return mixed_all.gather(2, idx[:, None, :].expand(M, mixed_all.shape[1], K - 1))


@torch.no_grad()
def _question_pass(net, pc, q_ids, q_valid, q_pos, q_img, q_split):
    """Run the question block once; returns final hidden [M,Tq,D] and per-layer caches."""
    xq = net.lm.embed_tokens(q_ids)
    rope_q = net.rope(xq, q_pos)
    from ablation.common import prefix_lm_mask
    mask_q = prefix_lm_mask(pc["p_valid"], q_valid, q_img, q_split)
    lens = q_valid.sum(1)
    vm = q_valid[..., None]
    caches = []
    for layer, ltype, c in zip(net.lm.layers, net.layer_types, pc["cache"]):
        hq = layer.input_layernorm(xq)
        if ltype == "linear_attention":
            m = layer.linear_attn
            _, S, tail = c
            K = m.conv_kernel_size
            w = m.conv1d.weight
            mixed_q, zq, beta_q, g_q = bb._gdn_mix(m, hq)
            beta_q = beta_q * vm.to(beta_q.dtype); g_q = g_q * vm.to(g_q.dtype)   # pads leave the state untouched
            mq = torch.cat([tail[q_img], mixed_q], dim=-1).to(w.dtype)
            conv_q = F.conv1d(mq, w, m.conv1d.bias, padding=0, groups=mq.shape[1])
            aq, Sq = bb._gdn_core(m, conv_q, zq, beta_q, g_q, S[q_img].contiguous())
            caches.append(["gdn", Sq, _gather_tail(mq, lens, K)])
        else:
            m = layer.self_attn
            _, kp, vp = c
            M, Tq, _ = hq.shape
            qq, kq, vq, gq = bb._qkv(m, hq, *rope_q)
            k_all = torch.cat([kp[q_img], kq], 2)
            v_all = torch.cat([vp[q_img], vq], 2)
            oq = F.scaled_dot_product_attention(qq, k_all, v_all, attn_mask=mask_q, scale=m.scaling, enable_gqa=True)
            aq = m.o_proj(oq.transpose(1, 2).reshape(M, Tq, -1) * torch.sigmoid(gq))
            caches.append(["attn", k_all, v_all])
        xq = xq + aq
        xq = xq + layer.mlp(layer.post_attention_layernorm(xq))
    kvalid = torch.cat([pc["p_valid"][q_img], q_valid], 1)
    return net.lm.norm(xq), caches, kvalid


@torch.no_grad()
def _step(net, caches, kvalid, tok, pos):
    """One causal token per row. tok [M], pos [M] (absolute 1-D position). Updates caches in place; returns hidden [M,D]."""
    M = tok.shape[0]
    x = net.lm.embed_tokens(tok[:, None])
    p3 = pos[None, :, None].expand(3, M, 1)
    rope = net.rope(x, p3)
    kvalid = torch.cat([kvalid, torch.ones(M, 1, dtype=torch.bool, device=tok.device)], 1)
    mask = kvalid[:, None, None, :]
    for layer, ltype, c in zip(net.lm.layers, net.layer_types, caches):
        h = layer.input_layernorm(x)
        if ltype == "linear_attention":
            m = layer.linear_attn
            _, S, tail = c
            w = m.conv1d.weight
            mixed, z, beta, g = bb._gdn_mix(m, h)
            mq = torch.cat([tail, mixed], -1).to(w.dtype)
            conv = F.conv1d(mq, w, m.conv1d.bias, padding=0, groups=mq.shape[1])
            a, S2 = bb._gdn_core(m, conv, z, beta, g, S.contiguous())
            c[1], c[2] = S2, mq[:, :, 1:]
        else:
            m = layer.self_attn
            q, k, v, gt = bb._qkv(m, h, *rope)
            c[1] = torch.cat([c[1], k], 2); c[2] = torch.cat([c[2], v], 2)
            o = F.scaled_dot_product_attention(q, c[1], c[2], attn_mask=mask, scale=m.scaling, enable_gqa=True)
            a = m.o_proj(o.transpose(1, 2).reshape(M, 1, -1) * torch.sigmoid(gt))
        x = x + a
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return net.lm.norm(x)[:, 0], kvalid


def _clone(caches):
    return [[t, a.clone(), b.clone()] for t, a, b in caches]


@torch.no_grad()
def fast_decode(net, fmt, batch, want_feats=False):
    """batch: collate_gen(with_answer=False) on device. Returns (res list like ar_decode(constrained, force_box), feats)."""
    dev = batch["q_ids"].device
    feats_img = net.image_features(batch["pixel_values"], batch["grid_thw"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pc = net.prefill(batch, feats_img)
        hq, caches, kvalid = _question_pass(net, pc, batch["q_ids"], batch["q_valid"], batch["q_pos"], batch["q_img"],
                                            batch["q_last"])
    M = hq.shape[0]
    ar = torch.arange(M, device=dev)
    qlen = batch["q_valid"].sum(1)
    pos0 = batch["q_pos"][0, :, 0] + qlen           # absolute position of the first answer token
    h0 = hq[ar, batch["q_last"]]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = net.logits(h0).float()
    logp = lg.log_softmax(-1)
    p = logp.exp()
    res = [dict() for _ in range(M)]
    pn = p[:, fmt.none]
    pc_ = p.clone(); pc_[:, fmt.none] = 0
    H = ent(pc_ / pc_.sum(-1, keepdim=True))
    top = logp.argmax(-1)
    pmax = pc_.max(-1).values
    v = torch.zeros_like(p, dtype=torch.bool); v[:, fmt.allowed([])] = True; v[:, fmt.none] = False
    pv = torch.where(v, p, torch.zeros_like(p))
    pmax_c, mass_c = pv.max(-1).values, pv.sum(-1)
    Hc = ent(pv / mass_c[:, None])
    for m in range(M):
        res[m].update(p_none=float(pn[m]), greedy_none=bool(top[m] == fmt.none), p_maxcoord=float(pmax[m]),
                      H_first=float(H[m]), p_maxcoord_c=float(pmax_c[m]), greedy_none_c=bool(pn[m] > pmax_c[m]),
                      p_none_c=float(pn[m] / (pn[m] + mass_c[m])), H_first_c=float(Hc[m]), coord_mass_valid=float(mass_c[m]))
    lp_none0 = logp[:, fmt.none]
    # "none" branch: log q(<box_end> | none)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hn, _ = _step(net, _clone(caches), kvalid, torch.full((M,), fmt.none, device=dev), pos0)
        lpn = net.logits(hn).float().log_softmax(-1)[:, fmt.box_end]
    for m in range(M):
        res[m]["lp_none"] = float(lp_none0[m] + lpn[m])
    # greedy, grammar-constrained, first token forced to a coordinate
    gen = [[] for _ in range(M)]
    lp = torch.zeros(M, device=dev)
    tl = torch.zeros(M, LMAX + 1, device=dev); tm = torch.zeros(M, LMAX + 1, device=dev)
    hs = [h0.float()]
    done = torch.zeros(M, dtype=torch.bool, device=dev)
    cur_logp = logp
    for step in range(LMAX + 1):
        if step > 0:
            if bool(done.all()) and not want_feats:
                break
        if step < LMAX and not bool(done.all()):
            cm = torch.zeros_like(cur_logp, dtype=torch.bool)
            for m in range(M):
                if not done[m]:
                    cm[m, fmt.allowed(gen[m])] = True
            if step == 0:
                cm[:, fmt.none] = False
            choose = cur_logp.masked_fill(~cm, -float("inf"))
            nxt = choose.argmax(-1)
            ch = cur_logp.gather(1, nxt[:, None])[:, 0]
            other = cur_logp.scatter(1, nxt[:, None], -float("inf")).max(-1).values
            act = ~done
            lp = lp + torch.where(act, ch, torch.zeros_like(ch))
            tl[act, step] = ch[act]; tm[act, step] = (ch - other)[act]
            nl = nxt.tolist()
            for m in range(M):
                if not done[m]:
                    gen[m].append(nl[m])
            done = done | (nxt == fmt.box_end)
            feed = torch.where(act, nxt, torch.full_like(nxt, fmt.box_end))
        else:
            break
        if bool(done.all()) and not want_feats:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, kvalid = _step(net, caches, kvalid, feed, pos0 + step)
            if want_feats:
                hs.append(h.float())
            cur_logp = net.logits(h).float().log_softmax(-1)
        if bool(done.all()):
            break
    lpl = lp.tolist()
    for m in range(M):
        res[m]["gen"] = gen[m]
        res[m]["lp_box"] = lpl[m]
    feats = None
    if want_feats:
        Hs = torch.stack(hs, 1)                                   # [M, 1+steps, D]: h_q, then hidden after feeding a_1..
        L = torch.tensor([len(g) for g in gen], device=dev)
        h_end = Hs[ar, L.clamp(max=Hs.shape[1] - 1)]
        j = torch.arange(Hs.shape[1], device=dev)[None]
        msk = ((j >= 1) & (j <= (L - 1)[:, None])).float()
        h_ans = (Hs * msk[..., None]).sum(1) / msk.sum(1, keepdim=True).clamp(min=1)
        h_ans = torch.where((L > 1)[:, None], h_ans, Hs[:, 0])
        feats = dict(h_q=Hs[:, 0].half().cpu(), h_ans=h_ans.half().cpu(), h_end=h_end.half().cpu(),
                     tok_lp=tl[:, :LMAX].cpu().numpy(), tok_mg=tm[:, :LMAX].cpu().numpy(), L=L.cpu().numpy())
    return res, feats
