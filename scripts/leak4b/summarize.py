"""results_raw.json (scripts/leak4b/run.py) -> results.json (aggregates + sanity + meta) and summary.md.

python scripts/leak4b/summarize.py outputs/leak4b
Aggregates are over 20 images (per block) or 20 images x 8 blocks (exactness). Keys in summary.md are the JSON paths in results.json.
"""
import json, sys

d = sys.argv[1] if len(sys.argv) > 1 else "outputs/leak4b"
RUNS = {"fla": "results_raw.json", "torch_gdn_fp32": "results_raw_torchgdn_fp32.json"}


def agg(vals, how):
    v = [float(x) for x in vals]
    return dict(mean=sum(v) / len(v), max=max(v), min=min(v), frac=sum(v) / len(v))[how]


HID = [("rel_norm", "mean"), ("rel_norm", "max"), ("rel_maxabs", "max"), ("min_cos", "min")]
RO = [("d_presence_logit", "max"), ("d_presence_prob", "mean"), ("d_presence_prob", "max"), ("center_kl", "mean"), ("center_kl", "max"),
      ("center_argmax_agree", "frac"), ("d_box", "max"), ("d_iou_prob", "max"), ("label_kl", "max"), ("vocab_kl", "mean"),
      ("vocab_kl", "max"), ("vocab_top1_agree", "frac")]


def hid_stats(rows, path):
    out = {}
    for m, how in HID:
        out[f"{m}_{how}"] = agg([r[path][m] if path else r[m] for r in rows], how)
    out["bitwise_equal_frac"] = agg([(r[path] if path else r)["bitwise_equal"] for r in rows], "frac")
    return out


def ro_stats(rows):
    return {f"{m}_{'frac' if how == 'frac' else how}": agg([r[m] for r in rows], how) for m, how in RO}


def aggregate(raw):
    DT = sorted({r["dtype"] for r in raw["exact"]}, reverse=True)  # fp32 first
    res = dict(meta=raw["meta"], sanity={}, exactness={}, leak={}, flex=raw.get("flex"))
    for dt in DT:
        s = [r for r in raw["sanity"] if r["dtype"] == dt]
        res["sanity"][dt] = {k: hid_stats([r[k] for r in s], None) for k in ["explicit_pos_vs_hf_pos", "dict_causal_vs_default"]}

    for dt in DT:
        for mode in ["causal", "block_bidir"]:
            rows = [r for r in raw["exact"] if r["dtype"] == dt and r["mode"] == mode]
            e = {}
            for refname in ["same_dtype_ref", "fp32_ref", "hf_bf16_vs_hf_fp32"]:
                rr = [r[refname] for r in rows if refname in r]
                if not rr:
                    continue
                e[refname] = dict(prefix=hid_stats(rr, "prefix"), block=hid_stats(rr, "block"),
                                  readout=ro_stats([x["readout"] for x in rr]))
                if "readout_hidden" in rr[0]:
                    e[refname]["readout_hidden"] = hid_stats(rr, "readout_hidden")
            res["exactness"][f"{dt}/{mode}"] = e

    for dt in DT:
        for v in ["blockmask", "blockmask_bidir", "nomask", "nomask_seqpos"]:
            rows = [r for r in raw["leak"] if r["dtype"] == dt and r["variant"] == v]
            pre = [r["prefix"] for r in rows if r["prefix"] is not None]
            L = dict(ref=rows[0]["ref"], prefix=dict(**hid_stats(pre, "vs_iso_block0"),
                                                      max_rel_norm_vs_all_iso=agg([p["max_rel_norm_vs_all_iso"] for p in pre], "max")),
                     per_block={})
            for k in range(8):
                rk = [r for r in rows if r["block"] == k]
                L["per_block"][str(k)] = dict(block_hidden=hid_stats(rk, "block_hidden"), readout_hidden=hid_stats(rk, "readout_hidden"),
                                              readout=ro_stats([r["readout"] for r in rk]))
            res["leak"][f"{dt}/{v}"] = L
    return res


ALL = {name: aggregate(json.load(open(f"{d}/{fn}"))) for name, fn in RUNS.items()}
json.dump(ALL, open(f"{d}/results.json", "w"), indent=1)
# ------------------------------------------------------------------ summary.md
f = lambda x: f"{x:.2e}" if (x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4)) else f"{x:.4f}"
m = ALL["fla"]["meta"]
L = []
L.append("# 4B shared-prefix exactness and block-mask leak (Glance v1.1)\n")
L.append(f"Model: {m['model']}. Data: {m['n_img']} RefCOCO-val images x {m['n_blocks']} referring expressions "
         f"(`scripts/leak4b/refcoco_val_20x8.jsonl`), v1.1 prompt layout, <=640 visual tokens (prefix 605-643 tokens, blocks 10-18 tokens). "
         f"{m['gpu']}, torch {m['torch']}, transformers {m['transformers']}, SDPA attention, TF32 off for torch matmuls. "
         "Two runs: `fla` = fla 0.6.0 Triton kernels for the Gated-DeltaNet layers (deployment path; fp32 and bf16), "
         "`torch_gdn_fp32` = the pure-PyTorch GDN reference implementation in both the engine and HF (fp32 only). "
         "Every number below is `results.json` at the key shown (`<run>.<section>.<dtype>/<mode>...`). "
         "Hidden states are final-normed last-layer states. Raw per-image/per-block rows: `results_raw*.json`.\n")
L.append("Metrics: `rel_norm` = ||x - ref|| / ||ref|| over the token span; `rel_maxabs` = max|x - ref| / max|ref|; `min_cos` = min per-token "
         "cosine; readout (v1.1 heads at the `<box_start>` readout token): presence probability sigmoid(logit / T_presence), "
         "center-patch distribution KL(ref || x) and argmax agreement, decoded box (normalized coords, max abs), "
         "28-way label distribution KL (typed head), full-vocab next-token KL(ref || x) and top-1 agreement.\n")

# headline
T, F = ALL["torch_gdn_fp32"], ALL["fla"]
ex = lambda R, key, ref="same_dtype_ref": R["exactness"][key][ref]
lk = lambda R, key: R["leak"][key]
mean_k = lambda R, key, path, met: [R["leak"][key]["per_block"][str(k)][path][met] for k in range(1, 8)]
L.append("## Headline\n")
e = ex(T, "fp32/block_bidir")
L.append(f"- Exactness, fp32, pure-torch GDN, deployed block_bidir mode: block hidden rel_norm max {f(e['block']['rel_norm_max'])} "
         f"(`torch_gdn_fp32.exactness.fp32/block_bidir.same_dtype_ref.block.rel_norm_max`), prefix {f(e['prefix']['rel_norm_max'])}, "
         f"|dP(present)| max {f(e['readout']['d_presence_prob_max'])}, vocab KL max {f(e['readout']['vocab_kl_max'])}.")
e = ex(F, "fp32/block_bidir")
L.append(f"- Same with fla kernels in fp32: block rel_norm max {f(e['block']['rel_norm_max'])} (`fla.exactness.fp32/block_bidir.same_dtype_ref.block.rel_norm_max`); "
         f"HF itself moves by {f(F['sanity']['fp32']['dict_causal_vs_default']['rel_norm_max'])} when only the SDPA kernel path changes "
         "(`fla.sanity.fp32.dict_causal_vs_default.rel_norm_max`), vs "
         f"{f(T['sanity']['fp32']['dict_causal_vs_default']['rel_norm_max'])} with torch GDN: the 1e-4 fp32 floor comes from fla's Triton kernels (not fp32-exact), not from the shared-prefix construction.")
e, n = ex(F, "bf16/block_bidir"), ex(F, "bf16/block_bidir", "hf_bf16_vs_hf_fp32")
e32 = ex(F, "bf16/block_bidir", "fp32_ref")
L.append(f"- bf16: engine vs HF bf16 block rel_norm max {f(e['block']['rel_norm_max'])}; against the fp32 reference the engine is at "
         f"{f(e32['block']['rel_norm_max'])} and HF bf16 itself at {f(n['block']['rel_norm_max'])} "
         f"(|dP| max {f(e32['readout']['d_presence_prob_max'])} vs {f(n['readout']['d_presence_prob_max'])}): the engine's bf16 deviation equals HF's own bf16 noise.")
p = lk(T, "fp32/blockmask")["prefix"]
L.append(f"- Leak harness prefix check (packed vs isolated prefix hidden states, fp32 torch GDN): rel_norm max {f(p['rel_norm_max'])} "
         f"(`torch_gdn_fp32.leak.fp32/blockmask.prefix.rel_norm_max`); bf16 fla {f(lk(F, 'bf16/blockmask')['prefix']['rel_norm_max'])}. Identical up to float noise.")
for key in ["fp32/blockmask", "fp32/nomask"]:
    b0 = lk(T, key)["per_block"]["0"]
    v = mean_k(T, key, "block_hidden", "rel_norm_mean")
    r = mean_k(T, key, "readout_hidden", "rel_norm_mean")
    dp = mean_k(T, key, "readout", "d_presence_prob_mean")
    dpx = max(mean_k(T, key, "readout", "d_presence_prob_max"))
    ca = mean_k(T, key, "readout", "center_argmax_agree_frac")
    L.append(f"- Leak `{key}` (fp32 torch GDN): block 0 (control) block-hidden rel_norm mean {f(b0['block_hidden']['rel_norm_mean'])}; "
             f"blocks 1-7 block-hidden rel_norm mean {min(v):.3f}-{max(v):.3f}, readout-token hidden {min(r):.3f}-{max(r):.3f}, "
             f"mean |dP(present)| {min(dp):.3f}-{max(dp):.3f} (max {dpx:.3f}), center-argmax agreement {min(ca):.2f}-{max(ca):.2f}.")
L.append("- The block-diagonal mask on the full-attention layers removes essentially none of the leak (blockmask vs nomask rows are within noise of each other): the Gated-DeltaNet "
         "recurrent state (plus its 4-tap conv window) carries the earlier blocks into block k, and the damage does not grow with k. Packing is not an exact substitute for per-block continuation "
         "on this hybrid; the shared-prefix engine (state copy per block) is.\n")
L.append("## 0. Harness sanity\n")
for run, R in ALL.items():
    for dt in R["sanity"]:
        for k in ["explicit_pos_vs_hf_pos", "dict_causal_vs_default"]:
            s_ = R["sanity"][dt][k]
            L.append(f"- `{run}.sanity.{dt}.{k}`: rel_norm max {f(s_['rel_norm_max'])}, bitwise-equal frac {f(s_['bitwise_equal_frac'])}")
L.append("\n(`explicit_pos_vs_hf_pos`: our M-RoPE positions passed explicitly vs HF computing its own (bitwise equal = positions identical); "
         "`dict_causal_vs_default`: HF with an explicit dict-form causal 4D mask vs HF's default causal path, i.e. the attention-kernel path alone.)\n")

L.append("## 1. Exactness of the shared-prefix engine vs HF forward on (prefix + block), 20 images x 8 blocks\n")
L.append("| key | prefix rel_norm max | prefix rel_maxabs max | block rel_norm mean / max | block rel_maxabs max | block min_cos | "
         "readout-token rel_norm max | dP(present) max | d presence logit max | center KL max | center argmax agree | box d max | label KL max | vocab KL max |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for run, R in ALL.items():
    for key, e in R["exactness"].items():
        for refname, x in e.items():
            pp, b, r = x["prefix"], x["block"], x["readout"]
            rh = f(x["readout_hidden"]["rel_norm_max"]) if "readout_hidden" in x else "-"
            L.append(f"| `{run}.exactness.{key}.{refname}` | {f(pp['rel_norm_max'])} | {f(pp['rel_maxabs_max'])} | {f(b['rel_norm_mean'])} / {f(b['rel_norm_max'])} | "
                     f"{f(b['rel_maxabs_max'])} | {b['min_cos_min']:.7f} | {rh} | {f(r['d_presence_prob_max'])} | {f(r['d_presence_logit_max'])} | "
                     f"{f(r['center_kl_max'])} | {f(r['center_argmax_agree_frac'])} | {f(r['d_box_max'])} | {f(r['label_kl_max'])} | {f(r['vocab_kl_max'])} |")
L.append("\n`same_dtype_ref` = engine vs HF in the same dtype; `fp32_ref` = engine bf16 vs HF fp32; `hf_bf16_vs_hf_fp32` = HF's own bf16 "
         "noise floor (no engine involved). `causal`: engine attn_mask=causal vs HF's default forward. `block_bidir` (deployed mode): "
         "vs HF with a dict mask (prefix causal, block bidirectional on the full-attention layers; GDN layers untouched). "
         "Center argmax agreement < 1 in bf16 occurs on near-flat center distributions (center KL max ~1e-2, same as HF bf16 vs fp32).\n")

L.append("## 2. Leak of block-diagonal packing (prefix + 8 blocks in one sequence) vs isolated (prefix + block k)\n")
L.append("Variants: `blockmask` = 4D mask on full-attention layers, prefix causal, block k sees prefix + itself (causal), positions restart "
         "after the prefix for each block; `blockmask_bidir` = same with bidirectional blocks (ref: isolated bidirectional); `nomask` = plain "
         "causal over the packed sequence, positions restart per block; `nomask_seqpos` = plain concatenation. Reference runs use the same "
         "attention kernel path as the packed run (`ref` below). Block 0 is the control (nothing precedes it but the prefix).\n")
L.append("Prefix check (packed prefix hidden states vs isolated prefix hidden states):\n")
for run, R in ALL.items():
    for key, x in R["leak"].items():
        pp = x["prefix"]
        L.append(f"- `{run}.leak.{key}.prefix`: rel_norm max {f(pp['rel_norm_max'])}, rel_maxabs max {f(pp['rel_maxabs_max'])}, "
                 f"min_cos {pp['min_cos_min']:.7f}, max over all 8 isolated runs {f(pp['max_rel_norm_vs_all_iso'])} (ref `{x['ref']}`)")
for run, R in ALL.items():
    for key, x in R["leak"].items():
        L.append(f"\n### `{run}.leak.{key}.per_block.<k>`\n")
        L.append("| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | "
                 "center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k in range(8):
            y = x["per_block"][str(k)]
            b, rh, r = y["block_hidden"], y["readout_hidden"], y["readout"]
            L.append(f"| {k} | {f(b['rel_norm_mean'])} / {f(b['rel_norm_max'])} | {f(rh['rel_norm_mean'])} | {f(r['d_presence_prob_mean'])} / "
                     f"{f(r['d_presence_prob_max'])} | {f(r['center_kl_mean'])} | {f(r['center_argmax_agree_frac'])} | {f(r['d_box_max'])} | "
                     f"{f(r['label_kl_max'])} | {f(r['vocab_kl_mean'])} | {f(r['vocab_top1_agree_frac'])} |")
L.append("\n## 3. FlexAttention\n")
L.append(f"- `fla.flex`: did not run. {F['flex']['error'].split('. Please')[0]} (transformers {m['transformers']}). No flex numbers are reported.\n")
L.append("## 4. Relation to the 27B test (`outputs/27b/bench/leak.json`)\n")
L.append("There the packed-vs-isolated prefix differed by 0.38 rel_norm, which cannot be leakage (the prefix is causal and comes first). "
         "Here the same check gives float noise (section 2), so the 27B prefix gap was a harness artifact, and the per-block numbers in this "
         "file supersede it for the 4B claim. Negative KL values of order 1e-8 are float rounding of a zero KL.\n")
open(f"{d}/summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
