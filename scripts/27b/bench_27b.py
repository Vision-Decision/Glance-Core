#!/usr/bin/env python
"""Standalone 27B memory/throughput + block-mask feasibility bench (phase 1).

Independent of the v0 training code. Loads Qwen3.5-family VLM (frozen vision tower),
attaches LoRA on LLM linear layers, runs fwd+bwd+AdamW with gradient checkpointing on a
synthetic batch shaped like the real data (1 image ~900 visual tokens + prompt + 8 question
blocks of 25 tokens), and tests a dict-form attention_mask with a shared-prefix block mask.

Usage (on VM):
  python bench_27b.py --mode throughput --quant bf16 --mbs 1,2,4 --maskmode block --out res.json
  python bench_27b.py --mode masktest --attn sdpa,flex --out mask.json
"""
import argparse, json, math, os, sys, time, traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.8-27B")
p.add_argument("--quant", default="bf16", choices=["bf16", "nf4", "fp8"])
p.add_argument("--mode", default="throughput", choices=["throughput", "masktest"])
p.add_argument("--mbs", default="1,2,4")
p.add_argument("--img", default="1024,896", help="H,W pixels; tokens=(H/32)*(W/32)")
p.add_argument("--prompt_len", type=int, default=40)
p.add_argument("--qblocks", type=int, default=8)
p.add_argument("--qlen", type=int, default=25)
p.add_argument("--lora_r", type=int, default=64)
p.add_argument("--steps", type=int, default=4, help="incl. 1 warmup")
p.add_argument("--maskmode", default="block", choices=["causal", "block"])
p.add_argument("--attn", default="sdpa")
p.add_argument("--no_gc", action="store_true")
p.add_argument("--out", default="/content/27b_out/bench.json")
args = p.parse_args()

dev = torch.device("cuda")
torch.manual_seed(0)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
RES = {"args": vars(args), "gpu": torch.cuda.get_device_name(0),
       "gpu_mem_gb": torch.cuda.get_device_properties(0).total_memory / 2**30, "runs": []}


def dump():
    with open(args.out, "w") as f:
        json.dump(RES, f, indent=1, default=str)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- environment
import transformers
from transformers import AutoConfig, AutoModelForImageTextToText

env = {"torch": torch.__version__, "transformers": transformers.__version__}
for mod in ["fla", "causal_conv1d", "peft", "bitsandbytes", "kernels"]:
    try:
        m = __import__(mod)
        env[mod] = getattr(m, "__version__", "ok")
    except Exception as e:
        env[mod] = f"missing ({type(e).__name__})"
RES["env"] = env
log("env", env)

# ---------------------------------------------------------------- model
cfg = AutoConfig.from_pretrained(args.model)
tc = cfg.text_config
t0 = time.time()
kw = dict(dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=args.attn.split(",")[0]
          if args.attn.split(",")[0] != "flex" else "sdpa")
if args.quant == "nf4":
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head", "in_proj_a", "in_proj_b"])
model_id = args.model
if args.quant == "fp8":
    model_id = args.model + "-FP8"
model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
model.config.use_cache = False
RES["load_s"] = time.time() - t0
RES["mem_after_load_gb"] = torch.cuda.memory_allocated() / 2**30
log(f"loaded {model_id} in {RES['load_s']:.0f}s, mem {RES['mem_after_load_gb']:.1f} GB")

for prm in model.parameters():
    prm.requires_grad_(False)

# LoRA targets: every Linear inside the LLM decoder layers with min(in,out) >= 512
targets, suffixes = [], set()
for n, m in model.named_modules():
    if ".language_model.layers." in n and isinstance(m, nn.Linear) and min(m.in_features, m.out_features) >= 512:
        targets.append(n)
        suffixes.add(".".join(n.split(".")[-2:]))
if not targets and args.quant == "nf4":
    import bitsandbytes as bnb
    for n, m in model.named_modules():
        if ".language_model.layers." in n and isinstance(m, bnb.nn.Linear4bit):
            targets.append(n); suffixes.add(".".join(n.split(".")[-2:]))
RES["lora_target_suffixes"] = sorted(suffixes)
log("lora targets", len(targets), sorted(suffixes))

from peft import LoraConfig, get_peft_model
model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                          target_modules=targets, bias="none"))
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
RES["lora_params_M"] = n_tr / 1e6
log(f"trainable LoRA params {n_tr/1e6:.1f}M")
if not args.no_gc:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # vision tower is frozen and runs without grad graph anyway; don't checkpoint it
base = model.base_model.model  # Qwen3_5ForConditionalGeneration
vlm = base.model               # Qwen3_5Model

# ---------------------------------------------------------------- synthetic batch
H, W = map(int, args.img.split(","))
ps, ms = cfg.vision_config.patch_size, cfg.vision_config.spatial_merge_size
gh, gw = H // ps, W // ps
n_img_tok = gh * gw // (ms * ms)
IMG, VS, VE = cfg.image_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id
Lp = args.prompt_len + 2 + n_img_tok  # shared prefix: prompt + <vs> image <ve>
L = Lp + args.qblocks * args.qlen
RES["seq"] = {"visual_tokens": n_img_tok, "prefix_len": Lp, "total_len": L}
log(f"visual tokens {n_img_tok}, prefix {Lp}, total {L}")


def make_batch(B, blocks=None):
    """blocks: list of block indices to include (default all)."""
    blocks = list(range(args.qblocks)) if blocks is None else blocks
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(1000, 100000, (args.prompt_len,), generator=g)
    qs = [torch.randint(1000, 100000, (args.qlen,), generator=g) for _ in range(args.qblocks)]
    seq = torch.cat([prompt, torch.tensor([VS]), torch.full((n_img_tok,), IMG), torch.tensor([VE])]
                    + [qs[k] for k in blocks])
    ids = seq[None].repeat(B, 1).to(dev)
    mm = (ids == IMG).int()
    pix = torch.randn(B * gh * gw, 3 * cfg.vision_config.temporal_patch_size * ps * ps,
                      generator=g).to(dev, torch.bfloat16)
    thw = torch.tensor([[1, gh, gw]] * B, device=dev)
    pos, _ = vlm.get_rope_index(ids, mm_token_type_ids=mm, image_grid_thw=thw)
    blk_id = torch.zeros(ids.shape[1], dtype=torch.long, device=dev)
    for j, k in enumerate(blocks):
        s = Lp + j * args.qlen
        blk_id[s:s + args.qlen] = j + 1
    # packed (shared-prefix) positions: every block restarts right after the prefix
    pos_packed = pos.clone()
    p0 = pos[:, :, Lp - 1:Lp] + 1
    for j in range(len(blocks)):
        s = Lp + j * args.qlen
        pos_packed[:, :, s:s + args.qlen] = p0 + torch.arange(args.qlen, device=dev)
    # answer-read positions: last token of each block
    read = torch.tensor([Lp + (j + 1) * args.qlen - 1 for j in range(len(blocks))], device=dev)
    return dict(ids=ids, mm=mm, pix=pix, thw=thw, pos=pos, pos_packed=pos_packed, blk=blk_id, read=read)


def block_allowed(blk):
    Lx = blk.shape[0]
    q = torch.arange(Lx, device=dev)
    causal = q[None, :] <= q[:, None]
    same = (blk[None, :] == 0) | (blk[None, :] == blk[:, None])
    return causal & same  # [q, k]


def additive(allowed, B):
    m = torch.zeros(allowed.shape, dtype=torch.bfloat16, device=dev)
    m.masked_fill_(~allowed, torch.finfo(torch.bfloat16).min)
    return m[None, None].expand(B, 1, *allowed.shape)


def run_forward(b, B, maskmode, full_mask=None):
    kwargs = dict(input_ids=b["ids"], pixel_values=b["pix"], image_grid_thw=b["thw"],
                  mm_token_type_ids=b["mm"], use_cache=False)
    if maskmode == "block":
        kwargs["position_ids"] = b["pos_packed"]
        fm = full_mask if full_mask is not None else additive(block_allowed(b["blk"]), B)
        kwargs["attention_mask"] = {"full_attention": fm, "linear_attention": None}
    else:
        kwargs["position_ids"] = b["pos"]
    return kwargs


# ---------------------------------------------------------------- throughput
def throughput():
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, fused=True)
    model.train()
    for B in [int(x) for x in args.mbs.split(",")]:
        r = {"quant": args.quant, "mbs": B, "maskmode": args.maskmode, "attn": args.attn, "gc": not args.no_gc}
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            b = make_batch(B)
            times = []
            for s in range(args.steps):
                torch.cuda.synchronize(); t = time.time()
                kw_ = run_forward(b, B, args.maskmode)
                # read logits only at the 8 answer positions (typed heads), not full vocab over L
                out = model(**kw_, logits_to_keep=b["read"])
                logits = out.logits.float()
                tgt = torch.randint(0, logits.shape[-1], logits.shape[:2], device=dev)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
                loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); times.append(time.time() - t)
                log(f"  mbs={B} step{s} {times[-1]:.2f}s loss={loss.item():.3f} "
                    f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB")
            st = sum(times[1:]) / max(1, len(times) - 1)
            r.update(ok=True, step_s=st, tok_per_s=B * L / st, records_per_s=B / st,
                     peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30,
                     peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30, step_times=times)
        except torch.cuda.OutOfMemoryError as e:
            r.update(ok=False, err="OOM", peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30)
            opt.zero_grad(set_to_none=True)
        except Exception as e:
            r.update(ok=False, err=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-3000:])
            opt.zero_grad(set_to_none=True)
        log(json.dumps({k: v for k, v in r.items() if k != "tb"}))
        RES["runs"].append(r); dump()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- mask test
def set_text_attn(impl):
    try:
        base.set_attn_implementation({"text_config": impl, "vision_config": "sdpa"})
        return "set_attn_implementation(dict)"
    except Exception as e:
        cfg.text_config._attn_implementation = impl
        base.config.text_config._attn_implementation = impl
        return f"direct config set ({type(e).__name__}: {str(e)[:120]})"


def hidden(kwargs):
    out = model(**kwargs, output_hidden_states=True, logits_to_keep=1)
    return out.hidden_states[-1][0].float()


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def masktest():
    model.eval()
    mt = {}
    full = make_batch(1)
    k = 3  # probe block index (0-based) > 0
    iso1 = make_batch(1, blocks=[0])
    isok = make_batch(1, blocks=[k])
    for impl in args.attn.split(","):
        rr = {}
        try:
            rr["how"] = set_text_attn("flex_attention" if impl == "flex" else impl)
            with torch.no_grad():
                h_c = hidden(run_forward(full, 1, "causal"))
                h_b = hidden(run_forward(full, 1, "block"))
                h_i1 = hidden(run_forward(iso1, 1, "causal"))
                h_ik = hidden(run_forward(isok, 1, "causal"))
            s1 = slice(Lp, Lp + args.qlen)
            sk = slice(Lp + k * args.qlen, Lp + (k + 1) * args.qlen)
            si = slice(Lp, Lp + args.qlen)
            rr["prefix_block_vs_causal_reldiff"] = rel(h_b[:Lp], h_c[:Lp])
            rr["blk0_block_vs_isolated_reldiff"] = rel(h_b[s1], h_i1[si])
            rr[f"blk{k}_block_vs_causal_reldiff(mask applied if >0)"] = rel(h_b[sk], h_c[sk])
            rr[f"blk{k}_block_vs_isolated_reldiff(leak via linear-attn)"] = rel(h_b[sk], h_ik[si])
            rr[f"blk{k}_causal_vs_isolated_reldiff(no-mask reference)"] = rel(h_c[sk], h_ik[si])
            # block-k logits agreement at its read position (what a typed head would see)
            lm = base.lm_head
            with torch.no_grad():
                pb = F.log_softmax(lm(h_b[sk][-1:].to(torch.bfloat16)).float(), -1)
                pi = F.log_softmax(lm(h_ik[si][-1:].to(torch.bfloat16)).float(), -1)
            rr[f"blk{k}_read_KL(block||isolated)"] = (pb.exp() * (pb - pi)).sum().item()
            # backward with GC + dict mask
            model.train()
            torch.cuda.reset_peak_memory_stats()
            b2 = make_batch(1)
            t = time.time()
            out = model(**run_forward(b2, 1, "block"), logits_to_keep=b2["read"])
            out.logits.float().logsumexp(-1).mean().backward()
            torch.cuda.synchronize()
            gn = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** .5
            model.zero_grad(set_to_none=True); model.eval()
            rr.update(backward_ok=True, fwd_bwd_s=time.time() - t, lora_grad_norm=gn,
                      peak_gb=torch.cuda.max_memory_allocated() / 2**30)
        except Exception as e:
            rr.update(ok=False, err=f"{type(e).__name__}: {str(e)[:800]}", tb=traceback.format_exc()[-3000:])
            model.zero_grad(set_to_none=True); model.eval()
        mt[impl] = rr
        log(impl, json.dumps({a: b for a, b in rr.items() if a != "tb"}, indent=1))
        RES["masktest"] = mt; dump()

    # flex with a real BlockMask (sparse path) instead of a dense 4D additive tensor
    if "flex" in args.attn:
        rr = {}
        try:
            from torch.nn.attention.flex_attention import create_block_mask
            set_text_attn("flex_attention")
            blk = full["blk"]

            def mask_mod(b, h, q, kv):
                return (kv <= q) & ((blk[kv] == 0) | (blk[kv] == blk[q]))
            bm = create_block_mask(mask_mod, 1, None, L, L, device=dev)
            with torch.no_grad():
                h_bm = hidden(run_forward(full, 1, "block", full_mask=bm))
                h_b = None
                set_text_attn("sdpa")
                h_b = hidden(run_forward(full, 1, "block"))
            rr["flex_BlockMask_vs_sdpa_dense_reldiff"] = rel(h_bm, h_b)
            set_text_attn("flex_attention")
            model.train()
            out = model(**run_forward(full, 1, "block", full_mask=bm), logits_to_keep=full["read"])
            out.logits.float().logsumexp(-1).mean().backward()
            model.zero_grad(set_to_none=True); model.eval()
            rr["backward_ok"] = True
        except Exception as e:
            rr.update(ok=False, err=f"{type(e).__name__}: {str(e)[:800]}", tb=traceback.format_exc()[-3000:])
        mt["flex_blockmask"] = rr
        log("flex_blockmask", json.dumps({a: b for a, b in rr.items() if a != "tb"}, indent=1))
        RES["masktest"] = mt; dump()


if args.mode == "throughput":
    throughput()
else:
    masktest()
dump()
log("DONE", args.out)
