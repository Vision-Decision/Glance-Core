"""4B exactness + block-mask leak measurement (Glance v1.1 merged weights = Qwen/Qwen3.5-4B + LoRA, plus v1.1 heads).

(a) exactness: shared-prefix engine (dv/model/backbone.py, prefix once, blocks continue from the prefix state)
    vs the reference HF forward on (prefix + block_k), per block, fp32 and bf16.
      causal      : engine attn_mask=causal  vs HF default forward (HF computes its own M-RoPE positions)
      block_bidir : engine attn_mask=block_bidir (the deployed mode) vs HF forward with a dict mask
                    (full-attention layers: prefix causal, block bidirectional; GDN layers unchanged)
    bf16 rows are also compared against the fp32 HF reference (noise floor: HF bf16 vs HF fp32).
(b) leak: prefix + 8 blocks packed into ONE sequence through the HF forward vs each block run alone after the prefix.
      blockmask       : full-attn layers see a block-diagonal mask (prefix causal; block k sees prefix + itself, causal);
                        positions restart after the prefix for every block. GDN layers run over the packed sequence.
      blockmask_bidir : same, bidirectional inside each block (deployed within-block mode); reference = isolated bidir.
      nomask          : plain causal over the packed sequence, positions restart per block.
      nomask_seqpos   : plain concatenation (causal, positions continue).
    Reference for each variant uses the same attention kernel path (dict mask <-> dict mask, default <-> default).
    Prefix check: packed prefix hidden states vs isolated prefix hidden states.

python scripts/leak4b/run.py --release /content/rel/v1.1/release --data refcoco_val_20x8.jsonl --img_root /content/imgs --out results_raw.json
"""
import argparse, json, os, sys, time
if "--no_fla" in sys.argv:  # pure-torch GDN path in BOTH the engine and HF (fla's Triton kernels are not fp32-exact)
    sys.modules["fla"] = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import copy
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from dv.config import Config
from dv.data.mixed import MixedDataset, collate_mixed
from dv.model.build import build
from dv.model.backbone import SharedPrefixLM

ap = argparse.ArgumentParser()
ap.add_argument("--release", default="/content/rel/v1.1/release")
ap.add_argument("--data", required=True)
ap.add_argument("--img_root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_img", type=int, default=20)
ap.add_argument("--flex", action="store_true")
ap.add_argument("--no_fla", action="store_true")
ap.add_argument("--dtypes", default="fp32,bf16")
ap.add_argument("--skip_leak", action="store_true")
a = ap.parse_args()
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

dc = json.load(open(f"{a.release}/dv_config.json"))
cfg = Config().update({k: v for k, v in dc["config"].items() if k in Config.__dataclass_fields__})
cfg = cfg.update(dict(model_id=f"{a.release}/merged-bf16", lora_r=0, grad_ckpt=False))
TP = dc.get("temperatures", {}).get("presence", 1.0)
net16, heads, proc = build(cfg, dev, lora=False)
heads.load_state_dict(load_file(f"{a.release}/heads.safetensors"))
heads.eval(); net16.eval()
hf16 = net16.hf
hf32 = copy.deepcopy(hf16).float().eval()
net32 = SharedPrefixLM(hf32, cfg).eval()
MODELS = {"bf16": (hf16, net16), "fp32": (hf32, net32)}
IMG_TOK = hf16.config.image_token_id
recs = [json.loads(l) for l in open(a.data)][: a.n_img]
ds = MixedDataset(None, a.img_root, proc, cfg, train=False, recs=recs)


def hcmp(x, ref):
    x, ref = x.float(), ref.float()
    d = x - ref
    return dict(rel_norm=(d.norm() / ref.norm()).item(), rel_maxabs=(d.abs().max() / ref.abs().max()).item(),
                maxabs=d.abs().max().item(), min_cos=F.cosine_similarity(x, ref, dim=-1).min().item(),
                bitwise_equal=bool(torch.equal(x, ref)))


@torch.no_grad()
def readout(hf, h_prefix, h_last, batch, img_m):
    """h_prefix [Lpv,D] valid prefix hidden (final-normed); h_last [M,D] readout-token hidden."""
    M = h_last.shape[0]
    vis = h_prefix[img_m][None]
    o = heads.box(h_last.float(), vis.float(), batch["vis_valid"], batch["centers"], torch.zeros(M, dtype=torch.long, device=dev))
    vocab = hf.lm_head(h_last.to(hf.lm_head.weight.dtype)).float().log_softmax(-1)
    return dict(pl=o["presence_logit"], cl=o["center_logits"].log_softmax(-1), box=o["box_pred"], iou=o["iou_logit"],
                lab=heads.typed(h_last.float()).log_softmax(-1), vocab=vocab)


def kl(p_log, q_log):  # KL(p || q) per row, p = reference
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def rcmp(x, ref, k):
    pr, px = torch.sigmoid(ref["pl"][k] / TP), torch.sigmoid(x["pl"][k] / TP)
    return dict(d_presence_logit=(x["pl"][k] - ref["pl"][k]).abs().item(), d_presence_prob=(px - pr).abs().item(),
                p_present_ref=pr.item(),
                center_kl=kl(ref["cl"][k], x["cl"][k]).item(), center_argmax_agree=bool(ref["cl"][k].argmax() == x["cl"][k].argmax()),
                d_box=(x["box"][k] - ref["box"][k]).abs().max().item(),
                d_iou_prob=(torch.sigmoid(x["iou"][k]) - torch.sigmoid(ref["iou"][k])).abs().item(),
                label_kl=kl(ref["lab"][k], x["lab"][k]).item(),
                vocab_kl=kl(ref["vocab"][k], x["vocab"][k]).item(),
                vocab_top1_agree=bool(ref["vocab"][k].argmax() == x["vocab"][k].argmax()))


def hf_fwd(hf, ids, pv, grid, pos=None, mask=None):
    kw = dict(input_ids=ids[None], pixel_values=pv, image_grid_thw=grid, mm_token_type_ids=(ids == IMG_TOK).int()[None])
    if pos is not None:
        kw["position_ids"] = pos[:, None, :]
    if mask is not None:
        kw["attention_mask"] = {"full_attention": mask[None, None], "linear_attention": None}
    return hf.model(**kw).last_hidden_state[0]


def seg_mask(Lp, lens, bidir):
    """bool [L,L], True = may attend. prefix causal; block k sees the prefix + itself (causal or bidirectional)."""
    L = Lp + sum(lens)
    m = torch.zeros(L, L, dtype=torch.bool, device=dev)
    m[:Lp, :Lp] = torch.ones(Lp, Lp, dtype=torch.bool, device=dev).tril()
    s = Lp
    for n in lens:
        m[s:s + n, :Lp] = True
        m[s:s + n, s:s + n] = True if bidir else torch.ones(n, n, dtype=torch.bool, device=dev).tril()
        s += n
    return m


raw = dict(exact=[], leak=[], sanity=[], flex=None)
t0 = time.time()
for ii in range(len(recs)):
    it = ds[ii]
    batch = collate_mixed([it], ds.tok)
    batch.pop("meta", None)
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    pvd = batch["p_valid"][0]
    pid, img_m, ppos = batch["p_ids"][0][pvd], batch["p_img_mask"][0][pvd], batch["p_pos"][:, 0][:, pvd]
    Lp = int(pid.shape[0])
    M = batch["q_ids"].shape[0]
    lens = [int(x) for x in batch["q_valid"].sum(-1)]
    qids = [batch["q_ids"][k][: lens[k]] for k in range(M)]
    qpos = [batch["q_pos"][:, k, : lens[k]] for k in range(M)]
    grid = batch["grid_thw"]
    ref32 = {}
    for dt in a.dtypes.split(","):
        hf, net = MODELS[dt]
        pv = batch["pixel_values"]
        with torch.no_grad():
            # ---------------- references: isolated prefix + block_k
            iso = {"causal_default": [], "causal_dict": [], "bidir_dict": []}
            for k in range(M):
                seq = torch.cat([pid, qids[k]])
                pos = torch.cat([ppos, qpos[k]], 1)
                iso["causal_default"].append(hf_fwd(hf, seq, pv, grid))
                iso["causal_dict"].append(hf_fwd(hf, seq, pv, grid, pos, seg_mask(Lp, [lens[k]], False)))
                iso["bidir_dict"].append(hf_fwd(hf, seq, pv, grid, pos, seg_mask(Lp, [lens[k]], True)))
                if k == 0:  # our positions == HF's own M-RoPE positions (explicit-pos default-mask run must be bitwise equal)
                    h_pos = hf_fwd(hf, seq, pv, grid, pos)
                    raw["sanity"].append(dict(img=ii, dtype=dt,
                                              explicit_pos_vs_hf_pos=hcmp(h_pos, iso["causal_default"][0]),
                                              dict_causal_vs_default=hcmp(iso["causal_dict"][0], iso["causal_default"][0])))
            iso_ro = {}
            for name, hs in iso.items():
                iso_ro[name] = readout(hf, hs[0][:Lp], torch.stack([hs[k][Lp + lens[k] - 1] for k in range(M)]), batch, img_m)
            if dt == "fp32":
                ref32 = dict(iso=iso, ro=iso_ro)
            # ---------------- (a) engine
            feats = net.image_features(batch["pixel_values"], grid)
            feats = [f.to(hf.dtype) for f in feats]
            for mode, rname in [("causal", "causal_default"), ("block_bidir", "bidir_dict")]:
                net.cfg = cfg.update(dict(attn_mask=mode))
                hp, hq = net(batch, feats)
                hpv = hp[0][pvd]
                ro = readout(hf, hpv, hq[torch.arange(M, device=dev), batch["q_last"]], batch, img_m)
                for k in range(M):
                    row = dict(img=ii, block=k, dtype=dt, mode=mode)
                    for refname, R, RO in [("same_dtype_ref", iso[rname], iso_ro[rname]),
                                           ("fp32_ref", ref32["iso"][rname], ref32["ro"][rname])]:
                        if dt == "fp32" and refname == "fp32_ref":
                            continue
                        row[refname] = dict(prefix=hcmp(hpv, R[k][:Lp]), block=hcmp(hq[k, : lens[k]], R[k][Lp:]),
                                            readout_hidden=hcmp(hq[k, lens[k] - 1], R[k][Lp + lens[k] - 1]),
                                            readout=rcmp(ro, RO, k))
                    if dt == "bf16":  # noise floor: HF bf16 reference vs HF fp32 reference
                        row["hf_bf16_vs_hf_fp32"] = dict(prefix=hcmp(iso[rname][k][:Lp], ref32["iso"][rname][k][:Lp]),
                                                         block=hcmp(iso[rname][k][Lp:], ref32["iso"][rname][k][Lp:]),
                                                         readout=rcmp(iso_ro[rname], ref32["ro"][rname], k))
                    raw["exact"].append(row)
            # ---------------- (b) packing
            if a.skip_leak:
                continue
            packed_ids = torch.cat([pid] + qids)
            pos_reset = torch.cat([ppos] + qpos, 1)
            q0 = int(qpos[0][0, 0])
            nq = sum(lens)
            pos_seq = torch.cat([ppos, (q0 + torch.arange(nq, device=dev)).view(1, -1).expand(3, -1)], 1)
            variants = {
                "blockmask": (pos_reset, seg_mask(Lp, lens, False), "causal_dict"),
                "blockmask_bidir": (pos_reset, seg_mask(Lp, lens, True), "bidir_dict"),
                "nomask": (pos_reset, None, "causal_default"),
                "nomask_seqpos": (pos_seq, None, "causal_default"),
            }
            for vname, (pos, mask, rname) in variants.items():
                h = hf_fwd(hf, packed_ids, pv, grid, pos, mask)
                starts = [Lp + sum(lens[:k]) for k in range(M)]
                ro = readout(hf, h[:Lp], torch.stack([h[starts[k] + lens[k] - 1] for k in range(M)]), batch, img_m)
                R, RO = iso[rname], iso_ro[rname]
                pre = dict(vs_iso_block0=hcmp(h[:Lp], R[0][:Lp]),
                           max_rel_norm_vs_all_iso=max(hcmp(h[:Lp], R[k][:Lp])["rel_norm"] for k in range(M)))
                for k in range(M):
                    sl = slice(starts[k], starts[k] + lens[k])
                    raw["leak"].append(dict(img=ii, block=k, dtype=dt, variant=vname, ref=rname, prefix=pre if k == 0 else None,
                                            block_hidden=hcmp(h[sl], R[k][Lp:]),
                                            readout_hidden=hcmp(h[starts[k] + lens[k] - 1], R[k][Lp + lens[k] - 1]),
                                            readout=rcmp(ro, RO, k)))
            # ---------------- flex (optional, one image, bf16)
            if a.flex and ii == 0 and dt == "bf16":
                try:
                    from torch.nn.attention.flex_attention import create_block_mask
                    bm = seg_mask(Lp, lens, False)
                    blk = create_block_mask(lambda b, h_, q, kv: bm[q, kv], None, None, bm.shape[0], bm.shape[0], device=dev)
                    hf.set_attn_implementation("flex_attention")
                    h = hf.model(input_ids=packed_ids[None], pixel_values=pv, image_grid_thw=grid,
                                 mm_token_type_ids=(packed_ids == IMG_TOK).int()[None], position_ids=pos_reset[:, None],
                                 attention_mask={"full_attention": blk, "linear_attention": None}).last_hidden_state[0]
                    hf.set_attn_implementation("sdpa")
                    starts = [Lp + sum(lens[:k]) for k in range(M)]
                    raw["flex"] = dict(ok=True, prefix=hcmp(h[:Lp], iso["causal_dict"][0][:Lp]),
                                       blocks=[hcmp(h[starts[k]:starts[k] + lens[k]], iso["causal_dict"][k][Lp:])["rel_norm"] for k in range(M)])
                except Exception as e:  # noqa
                    try:
                        hf.set_attn_implementation("sdpa")
                    except Exception:
                        pass
                    raw["flex"] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:400]}")
                print("flex", raw["flex"], flush=True)
    s = [r for r in raw["sanity"] if r["img"] == ii]
    lk = [r for r in raw["leak"] if r["img"] == ii and r["variant"] == "blockmask" and r["dtype"] == "fp32"]
    if not lk:
        ex = [r["same_dtype_ref"] for r in raw["exact"] if r["img"] == ii]
        print(f"[{ii}] {time.time() - t0:.0f}s sanity={[(x['dtype'], x['dict_causal_vs_default']['rel_norm']) for x in s]} "
              f"exact prefix/block rel_norm max={max(e['prefix']['rel_norm'] for e in ex):.2e}/{max(e['block']['rel_norm'] for e in ex):.2e}", flush=True)
        continue
    print(f"[{ii}] {time.time() - t0:.0f}s Lp={Lp} lens={lens} sanity={[(x['dtype'], round(x['explicit_pos_vs_hf_pos']['rel_norm'], 8), round(x['dict_causal_vs_default']['rel_norm'], 6)) for x in s]} "
          f"prefix_fp32={lk[0]['prefix']['vs_iso_block0']['rel_norm']:.2e} leak_fp32={[round(r['block_hidden']['rel_norm'], 4) for r in lk]}", flush=True)

raw["meta"] = dict(release=a.release, model="Glance v1.1 merged-bf16 (Qwen/Qwen3.5-4B + v1.1 LoRA) + v1.1 heads",
                   n_img=len(recs), n_blocks=8, presence_temperature=TP, torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(), tf32=False, gdn="torch" if a.no_fla else "fla", seconds=time.time() - t0)
import transformers
raw["meta"]["transformers"] = transformers.__version__
try:
    import fla
    raw["meta"]["fla"] = fla.__version__
except Exception:
    raw["meta"]["fla"] = None
json.dump(raw, open(a.out, "w"))
print("DONE", a.out)
