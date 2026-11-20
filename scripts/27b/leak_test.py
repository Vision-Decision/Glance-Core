"""Kernel-matched leakage test: packed shared-prefix blocks with a block mask on full-attn layers
vs each block run alone after the prefix. Both runs use the dict-form mask (same masked sdpa kernel),
so any difference on block k>0 = cross-block leakage through the Gated-DeltaNet recurrence/conv."""
import sys, json, torch, torch.nn.functional as F
sys.argv = ["x", "--mode", "masktest", "--out", "/content/27b_out/leak.json"]
exec(open("/content/27b/bench_27b.py").read().split("if args.mode ==")[0])
model.eval()
full = make_batch(1)
def hs(b, mode, fm=None):
    kw = run_forward(b, 1, "block", full_mask=fm)
    with torch.no_grad():
        return model(**kw, output_hidden_states=True, logits_to_keep=1).hidden_states[-1][0].float()
hb = hs(full, "block")
res = {}
for k in range(args.qblocks):
    iso = make_batch(1, blocks=[k])
    Li = iso["ids"].shape[1]
    hi = hs(iso, "block", additive(torch.ones(Li, Li, dtype=torch.bool, device=dev).tril(), 1))
    sk = slice(Lp + k * args.qlen, Lp + (k + 1) * args.qlen); si = slice(Lp, Lp + args.qlen)
    pb = F.log_softmax(base.lm_head(hb[sk][-1:].bfloat16()).float(), -1)
    pi = F.log_softmax(base.lm_head(hi[si][-1:].bfloat16()).float(), -1)
    res[k] = dict(prefix_reldiff=((hb[:Lp]-hi[:Lp]).norm()/hi[:Lp].norm()).item(),
                  block_reldiff=((hb[sk]-hi[si]).norm()/hi[si].norm()).item(),
                  read_KL=(pb.exp()*(pb-pi)).sum().item(),
                  top1_agree=bool(pb.argmax()==pi.argmax()))
    print(k, res[k], flush=True)
json.dump(res, open("/content/27b_out/leak.json", "w"), indent=1)
