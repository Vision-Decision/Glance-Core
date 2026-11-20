"""Incremental (KV-cached) decoding for the ablation AR heads, plus HF-style beam search with an optional
"refined" output space.

Equivalence with ablation.eval_ab.ar_decode: there, every decode step re-runs the whole question block
(question tokens bidirectional among themselves, answer tokens causal, question tokens never see answer tokens).
Because question tokens never see answer tokens, their states do not change across steps, so caching
  * full-attention layers: K/V of [prefix (all valid keys), question tokens, answers so far]
  * Gated-DeltaNet layers: the recurrent state after the last processed token + the last (k-1) conv inputs
and feeding one new token per step gives the same logits (up to bf16 kernel noise). `ablation/beam_ab.py --check`
verifies this against the rows of the constrained greedy eval.

GDN states must be taken at each block's own last token, so question blocks are started in groups of equal length
(no right padding inside a group); afterwards all caches have fixed per-row shapes (attention keys are padded with
an invalid-key mask).

Beam search follows transformers' (5.x, `GenerationMixin._beam_search`) semantics:
  * scores are full-vocabulary log-probs (log_softmax over the whole vocab), grammar-constrained by masking
    (= `prefix_allowed_tokens_fn`); "none" is an allowed first token, so "none <|box_end|>" competes as a hypothesis;
  * at every step the top 2B continuations are taken; continuations that emit <|box_end|> become finished
    hypotheses if they rank within the top B, the others refill the B live beams;
  * finished score = sum_logprobs / gen_len**length_penalty, gen_len = generated tokens incl. <|box_end|>
    (transformers 5.23: `topk_log_probs / ((cur_len + 1 - decoder_prompt_len) ** length_penalty)`);
  * done (early_stopping=False) when B hypotheses are finished and the worst of them >= best live sum_logprob /
    cur_len**length_penalty; live beams left at max_len are finalised as hypotheses.
Refinement (Prop. 2): `split` = {token_id: k}. Each listed token is replaced by k equivalent surface forms, each
with probability q(token)/k (log-prob - log k). The next model input is the original token (a sub-form carries no
extra information), so the refined model is a true refinement r'(s') = r(phi(s')) / prod k of the same trained
distribution. Beam search then runs over the refined space (copies are distinct hypotheses).
"""
import math
import torch
import torch.nn.functional as F

from ablation.dvcore import backbone as bb
from ablation.common import prefix_lm_mask


class IncCache:
    """Per-row decoding state. layers[i] = ('gdn', S, tail) or ('attn', k, v); kmask [N, L] valid keys."""

    def __init__(self, layers, kmask, pos):
        self.layers, self.kmask, self.pos = layers, kmask, pos

    def select(self, idx):
        lay = []
        for c in self.layers:
            lay.append((c[0], c[1].index_select(0, idx), c[2].index_select(0, idx)))
        return IncCache(lay, self.kmask.index_select(0, idx), self.pos.index_select(0, idx))

    @staticmethod
    def cat(caches):
        L = max(c.kmask.shape[1] for c in caches)
        lay = []
        for i, c0 in enumerate(caches[0].layers):
            if c0[0] == "gdn":
                lay.append(("gdn", torch.cat([c.layers[i][1] for c in caches]), torch.cat([c.layers[i][2] for c in caches])))
            else:
                ks, vs = [], []
                for c in caches:
                    k, v = c.layers[i][1], c.layers[i][2]
                    pad = L - k.shape[2]
                    if pad:
                        k = F.pad(k, (0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, pad))
                    ks.append(k); vs.append(v)
                lay.append(("attn", torch.cat(ks), torch.cat(vs)))
        km = torch.cat([F.pad(c.kmask, (0, L - c.kmask.shape[1]), value=False) for c in caches])
        return IncCache(lay, km, torch.cat([c.pos for c in caches]))


@torch.no_grad()
def start_blocks(net, pc, batch):
    """Run every question block once (from the prefix cache of net.prefill). Returns (IncCache over questions in
    batch order, logits [M,V] at <box_start>)."""
    q_valid = batch["q_valid"]
    lens = q_valid.sum(1)
    M = lens.shape[0]
    dev = q_valid.device
    groups = {}
    for m, l in enumerate(lens.tolist()):
        groups.setdefault(l, []).append(m)
    caches, logits, order = [], [], []
    for T, idx in groups.items():
        sel = torch.tensor(idx, device=dev)
        ids = batch["q_ids"][sel, :T]
        qv = torch.ones_like(ids, dtype=torch.bool)
        pos = batch["q_pos"][:, sel, :T]
        q_img = batch["q_img"][sel]
        xq = net.embed(ids)
        rope_q = net.rope(xq, pos)
        mask_q = prefix_lm_mask(pc["p_valid"], qv, q_img, torch.full((len(idx),), T - 1, device=dev))
        lay = []
        for layer, ltype, c in zip(net.lm.layers, net.layer_types, pc["cache"]):
            hq = layer.input_layernorm(xq)
            if ltype == "linear_attention":
                m = layer.linear_attn
                _, S, tail = c
                w = m.conv1d.weight
                K = m.conv_kernel_size
                mixed_q, zq, beta_q, g_q = bb._gdn_mix(m, hq)
                mq = torch.cat([tail[q_img], mixed_q], dim=-1).to(w.dtype)
                conv_q = F.conv1d(mq, w, m.conv1d.bias, padding=0, groups=mq.shape[1])
                aq, S2 = bb._gdn_core(m, conv_q, zq, beta_q, g_q, S[q_img].contiguous())
                lay.append(("gdn", S2, mq[:, :, mq.shape[-1] - (K - 1):].contiguous()))
            else:
                m = layer.self_attn
                _, kp, vp = c
                n, _, _ = hq.shape
                qq, kq, vq, gq = bb._qkv(m, hq, *rope_q)
                k_all = torch.cat([kp[q_img], kq], 2)
                v_all = torch.cat([vp[q_img], vq], 2)
                oq = F.scaled_dot_product_attention(qq, k_all, v_all, attn_mask=mask_q, scale=m.scaling, enable_gqa=True)
                aq = m.o_proj(oq.transpose(1, 2).reshape(n, T, -1) * torch.sigmoid(gq))
                lay.append(("attn", k_all, v_all))
            xq = xq + aq
            xq = xq + layer.mlp(layer.post_attention_layernorm(xq))
        h = net.lm.norm(xq[:, -1])
        logits.append(net.logits(h).float())
        km = torch.cat([pc["p_valid"][q_img], qv], 1)
        caches.append(IncCache(lay, km, pos[0, :, -1] + 1))
        order += idx
    cache = IncCache.cat(caches)
    lg = torch.cat(logits)
    inv = torch.empty(M, dtype=torch.long, device=dev)
    inv[torch.tensor(order, device=dev)] = torch.arange(M, device=dev)
    return cache.select(inv), lg[inv]


@torch.no_grad()
def step(net, cache, tok):
    """Feed one token per row. tok [N] long. Returns (new cache, logits [N,V] fp32)."""
    N = tok.shape[0]
    x = net.embed(tok[:, None])
    pos3 = cache.pos[None, :, None].expand(3, N, 1)
    rope = net.rope(x, pos3)
    km = torch.cat([cache.kmask, torch.ones(N, 1, dtype=torch.bool, device=tok.device)], 1)
    amask = km[:, None, None, :]
    lay = []
    for layer, ltype, c in zip(net.lm.layers, net.layer_types, cache.layers):
        h = layer.input_layernorm(x)
        if ltype == "linear_attention":
            m = layer.linear_attn
            _, S, tail = c
            w = m.conv1d.weight
            mixed, z, beta, g = bb._gdn_mix(m, h)
            mq = torch.cat([tail, mixed.to(tail.dtype)], -1).to(w.dtype)
            conv = F.conv1d(mq, w, m.conv1d.bias, padding=0, groups=mq.shape[1])
            a, S2 = bb._gdn_core(m, conv, z, beta, g, S.contiguous())
            lay.append(("gdn", S2, mq[:, :, 1:].contiguous()))
        else:
            m = layer.self_attn
            _, kc, vc = c
            q, k, v, gate = bb._qkv(m, h, *rope)
            kc = torch.cat([kc, k], 2); vc = torch.cat([vc, v], 2)
            o = F.scaled_dot_product_attention(q, kc, vc, attn_mask=amask, scale=m.scaling, enable_gqa=True)
            a = m.o_proj(o.transpose(1, 2).reshape(N, 1, -1) * torch.sigmoid(gate))
            lay.append(("attn", kc, vc))
        x = x + a
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    lg = net.logits(net.lm.norm(x[:, 0])).float()
    return IncCache(lay, km, cache.pos + 1), lg


# ------------------------------------------------------------------------------------------------ grammar helpers
def grammar_vocab(fmt):
    """All token ids the answer grammar can ever emit (subset of the vocab used for candidate scoring)."""
    toks = set(fmt.allowed([])) | {fmt.box_end}
    if hasattr(fmt, "digit"):
        toks |= set(fmt.digit) | {fmt.comma}
    if hasattr(fmt, "N"):
        toks |= set(range(fmt.V0, fmt.V0 + fmt.N))
    if hasattr(fmt, "B"):
        toks |= set(range(fmt.V0, fmt.V0 + 2 * fmt.B))
    return sorted(toks)


@torch.no_grad()
def greedy(net, fmt, cache, lg0, force_box=True, on_first_done=None):
    """Constrained greedy with incremental decoding (for the equivalence check and latency).
    on_first_done: callable invoked once, when the first row emits <|box_end|> (time-to-first-answer).
    Returns list of (gen tokens, sum logprob)."""
    N = lg0.shape[0]
    gen = [[] for _ in range(N)]
    lp = [0.0] * N
    done = [False] * N
    lg = lg0
    rows = torch.arange(N, device=lg0.device)
    for t in range(fmt.max_len):
        logp = lg.log_softmax(-1)
        mask = torch.zeros_like(logp, dtype=torch.bool)
        act = [i for i in range(len(rows)) if not done[int(rows[i])]]
        for i in range(len(rows)):
            mask[i, fmt.allowed(gen[int(rows[i])])] = True
        if t == 0 and force_box:
            mask[:, fmt.none] = False
        nxt = logp.masked_fill(~mask, -float("inf")).argmax(-1)
        for i in range(len(rows)):
            n = int(rows[i]); tk = int(nxt[i])
            gen[n].append(tk); lp[n] += float(logp[i, tk])
            if tk in fmt.stop:
                done[n] = True
                if on_first_done is not None:
                    on_first_done(); on_first_done = None
        keep = [i for i in range(len(rows)) if not done[int(rows[i])]]
        if not keep:
            break
        ki = torch.tensor(keep, device=lg0.device)
        cache = cache.select(ki); rows = rows[ki]
        cache, lg = step(net, cache, nxt[ki])
    return list(zip(gen, lp))


@torch.no_grad()
def beam_search(net, fmt, cache, lg0, B=4, length_penalty=0.0, split=None, n_return=None, none_lp=None):
    """HF-style constrained beam search for every row of (cache, lg0) (= one question each).
    split: {token_id: k} refinement (see module doc). Returns per question a list of finished hypotheses sorted by
    score: dict(tokens, sum_lp, score). sum_lp is the log-prob in the (refined) output space."""
    dev = lg0.device
    Q = lg0.shape[0]
    vocab = grammar_vocab(fmt)
    vt = torch.tensor(vocab, device=dev)
    cols, pen = [], []                      # expanded candidate columns: (token, copy), log-prob penalty -log k
    for t in vocab:
        k = (split or {}).get(t, 1)
        for _ in range(k):
            cols.append(t); pen.append(-math.log(k))
    col_pen = torch.tensor(pen, device=dev)
    vpos = {t: i for i, t in enumerate(vocab)}
    col_src = torch.tensor([vpos[t] for t in cols], device=dev)
    C = len(cols)
    amemo = {}

    def allowed_cols(g):
        key = tuple(g)
        if key not in amemo:
            a = set(fmt.allowed(list(g)))
            amemo[key] = torch.tensor([t in a for t in cols], device=dev)
        return amemo[key]

    hyps = [[] for _ in range(Q)]           # finished: (score, sum_lp, tokens)
    meta = [(q, 0.0, []) for q in range(Q)]  # cache row -> (question, sum_lp, tokens); question-major
    if none_lp is not None:
        for q in range(Q):
            hyps[q].append((none_lp[q] / (2 ** length_penalty), none_lp[q], [fmt.none, fmt.box_end]))
    lg = lg0
    for t in range(fmt.max_len + 1):
        logp = lg.log_softmax(-1)[:, vt][:, col_src] + col_pen                    # [R, C]
        amask = torch.stack([allowed_cols(m[2]) for m in meta])
        base = torch.tensor([m[1] for m in meta], device=dev, dtype=logp.dtype)
        if none_lp is not None and t == 0:
            amask = amask & torch.tensor([c != fmt.none for c in cols], device=dev)[None]
        sc = (base[:, None] + logp).masked_fill(~amask, -float("inf")).cpu()
        cur_len = t + 1
        nxt_rows, nxt_tok, nxt_meta = [], [], []
        r0 = 0
        R = len(meta)
        while r0 < R:
            q = meta[r0][0]
            r1 = r0
            while r1 < R and meta[r1][0] == q:
                r1 += 1
            block = sc[r0:r1].reshape(-1)
            kk = min(2 * B, int(torch.isfinite(block).sum()))
            live = []
            if kk:
                top = torch.topk(block, kk)
                for rank, (s, ix) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
                    bi, ci = divmod(ix, C)
                    tok = cols[ci]
                    toks = meta[r0 + bi][2] + [tok]
                    if tok in fmt.stop:
                        if rank < B:
                            hyps[q].append((s / (len(toks) ** length_penalty), s, toks))
                    elif len(live) < B:
                        live.append((r0 + bi, tok, s, toks))
                    if len(live) >= B:
                        break
            finished = False
            if len(hyps[q]) >= B:
                hyps[q].sort(key=lambda h: -h[0])
                del hyps[q][B:]
                best_live = max((x[2] for x in live), default=-float("inf"))
                finished = hyps[q][-1][0] >= best_live / (cur_len ** length_penalty)
            if t == fmt.max_len:            # finalise beams still alive at max length
                for _, _, s, toks in live:
                    hyps[q].append((s / (max(len(toks), 1) ** length_penalty), s, toks))
                finished = True
            if not finished:
                for r, tok, s, toks in live:
                    nxt_rows.append(r); nxt_tok.append(tok); nxt_meta.append((q, s, toks))
            r0 = r1
        if not nxt_rows:
            break
        meta = nxt_meta
        cache = cache.select(torch.tensor(nxt_rows, device=dev))
        cache, lg = step(net, cache, torch.tensor(nxt_tok, device=dev))
    out = []
    for q in range(Q):
        h = sorted(hyps[q], key=lambda x: -x[0])[: (n_return or B)]
        out.append([dict(tokens=x[2], sum_lp=x[1], score=x[0]) for x in h])
    return out
