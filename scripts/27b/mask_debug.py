"""Debug: why does the prefix differ between causal and block-mask runs?
Compares, on prefix tokens, final hidden states for
  A) attention_mask=None (HF causal, sdpa is_causal fast path)
  B) dict mask with a *pure causal* 4D additive mask (same math as A, masked sdpa kernel)
  C) dict mask with the shared-prefix block mask
and also reports per-layer divergence and per-token concentration.
Run after bench_27b.py (reuses its helpers via exec of the top half)."""
import sys, torch
sys.argv = ["x", "--mode", "masktest", "--out", "/content/27b_out/mask_debug.json"]
src = open("/content/27b/bench_27b.py").read().split("if args.mode ==")[0]
exec(src)
model.eval()
b = make_batch(1)
Lx = b["ids"].shape[1]
causal_allowed = torch.ones(Lx, Lx, dtype=torch.bool, device=dev).tril()


def hs(kw):
    with torch.no_grad():
        o = model(**kw, output_hidden_states=True, logits_to_keep=1)
    return [h[0].float() for h in o.hidden_states]


A = hs(run_forward(b, 1, "causal"))
kB = run_forward(b, 1, "block", full_mask=additive(causal_allowed, 1)); kB["position_ids"] = b["pos"]
B_ = hs(kB)
C = hs(run_forward(b, 1, "block"))
A2 = hs(run_forward(b, 1, "causal"))  # determinism check
out = {}
for name, X in [("A_rerun", A2), ("B_dict_causal", B_), ("C_dict_block", C)]:
    per_layer = [((X[i][:Lp] - A[i][:Lp]).norm() / A[i][:Lp].norm()).item() for i in range(len(A))]
    d = (X[-1][:Lp] - A[-1][:Lp]).norm(dim=-1) / A[-1][:Lp].norm(dim=-1)
    out[name] = dict(final_prefix_reldiff=per_layer[-1], prelast_prefix_reldiff=per_layer[-2],
                     layers_1_4_8_16_32=[round(per_layer[i], 4) for i in (1, 4, 8, 16, 32)],
                     per_token_median=d.median().item(), per_token_max=d.max().item(),
                     worst_tokens=d.topk(5).indices.tolist())
    print(name, out[name], flush=True)
norms = A[-1][:Lp].norm(dim=-1)
print("final-hidden norm: median", norms.median().item(), "max", norms.max().item(), "argmax", norms.argmax().item())
pre = A[-2][:Lp].norm(dim=-1)
print("pre-norm hidden norm: median", pre.median().item(), "max", pre.max().item(), "argmax", pre.argmax().item())
import json; json.dump(out, open("/content/27b_out/mask_debug.json", "w"), indent=1)
