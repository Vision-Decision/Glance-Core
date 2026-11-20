# 4B shared-prefix exactness and block-mask leak (Glance v1.1)

Model: Glance v1.1 merged-bf16 (Qwen/Qwen3.5-4B + v1.1 LoRA) + v1.1 heads. Data: 20 RefCOCO-val images x 8 referring expressions (`scripts/leak4b/refcoco_val_20x8.jsonl`), v1.1 prompt layout, <=640 visual tokens (prefix 605-643 tokens, blocks 10-18 tokens). NVIDIA RTX PRO 6000 Blackwell Server Edition, torch 2.14.1+cu130, transformers 5.23.0, SDPA attention, TF32 off for torch matmuls. Two runs: `fla` = fla 0.6.0 Triton kernels for the Gated-DeltaNet layers (deployment path; fp32 and bf16), `torch_gdn_fp32` = the pure-PyTorch GDN reference implementation in both the engine and HF (fp32 only). Every number below is `results.json` at the key shown (`<run>.<section>.<dtype>/<mode>...`). Hidden states are final-normed last-layer states. Raw per-image/per-block rows: `results_raw*.json`.

Metrics: `rel_norm` = ||x - ref|| / ||ref|| over the token span; `rel_maxabs` = max|x - ref| / max|ref|; `min_cos` = min per-token cosine; readout (v1.1 heads at the `<box_start>` readout token): presence probability sigmoid(logit / T_presence), center-patch distribution KL(ref || x) and argmax agreement, decoded box (normalized coords, max abs), 28-way label distribution KL (typed head), full-vocab next-token KL(ref || x) and top-1 agreement.

## Headline

- Exactness, fp32, pure-torch GDN, deployed block_bidir mode: block hidden rel_norm max 9.93e-07 (`torch_gdn_fp32.exactness.fp32/block_bidir.same_dtype_ref.block.rel_norm_max`), prefix 6.69e-07, |dP(present)| max 1.88e-06, vocab KL max 4.53e-07.
- Same with fla kernels in fp32: block rel_norm max 2.15e-04 (`fla.exactness.fp32/block_bidir.same_dtype_ref.block.rel_norm_max`); HF itself moves by 5.20e-05 when only the SDPA kernel path changes (`fla.sanity.fp32.dict_causal_vs_default.rel_norm_max`), vs 3.66e-07 with torch GDN: the 1e-4 fp32 floor comes from fla's Triton kernels (not fp32-exact), not from the shared-prefix construction.
- bf16: engine vs HF bf16 block rel_norm max 0.0076; against the fp32 reference the engine is at 0.0189 and HF bf16 itself at 0.0200 (|dP| max 0.0111 vs 0.0175): the engine's bf16 deviation equals HF's own bf16 noise.
- Leak harness prefix check (packed vs isolated prefix hidden states, fp32 torch GDN): rel_norm max 6.24e-07 (`torch_gdn_fp32.leak.fp32/blockmask.prefix.rel_norm_max`); bf16 fla 0.0064. Identical up to float noise.
- Leak `fp32/blockmask` (fp32 torch GDN): block 0 (control) block-hidden rel_norm mean 5.48e-07; blocks 1-7 block-hidden rel_norm mean 0.156-0.177, readout-token hidden 0.044-0.087, mean |dP(present)| 0.006-0.037 (max 0.274), center-argmax agreement 0.55-0.85.
- Leak `fp32/nomask` (fp32 torch GDN): block 0 (control) block-hidden rel_norm mean 5.66e-07; blocks 1-7 block-hidden rel_norm mean 0.161-0.175, readout-token hidden 0.048-0.086, mean |dP(present)| 0.004-0.024 (max 0.269), center-argmax agreement 0.65-0.80.
- The block-diagonal mask on the full-attention layers removes essentially none of the leak (blockmask vs nomask rows are within noise of each other): the Gated-DeltaNet recurrent state (plus its 4-tap conv window) carries the earlier blocks into block k, and the damage does not grow with k. Packing is not an exact substitute for per-block continuation on this hybrid; the shared-prefix engine (state copy per block) is.

## 0. Harness sanity

- `fla.sanity.fp32.explicit_pos_vs_hf_pos`: rel_norm max 0.0000, bitwise-equal frac 1.0000
- `fla.sanity.fp32.dict_causal_vs_default`: rel_norm max 5.20e-05, bitwise-equal frac 0.0000
- `fla.sanity.bf16.explicit_pos_vs_hf_pos`: rel_norm max 0.0000, bitwise-equal frac 1.0000
- `fla.sanity.bf16.dict_causal_vs_default`: rel_norm max 0.0056, bitwise-equal frac 0.0000
- `torch_gdn_fp32.sanity.fp32.explicit_pos_vs_hf_pos`: rel_norm max 0.0000, bitwise-equal frac 1.0000
- `torch_gdn_fp32.sanity.fp32.dict_causal_vs_default`: rel_norm max 3.66e-07, bitwise-equal frac 0.0000

(`explicit_pos_vs_hf_pos`: our M-RoPE positions passed explicitly vs HF computing its own (bitwise equal = positions identical); `dict_causal_vs_default`: HF with an explicit dict-form causal 4D mask vs HF's default causal path, i.e. the attention-kernel path alone.)

## 1. Exactness of the shared-prefix engine vs HF forward on (prefix + block), 20 images x 8 blocks

| key | prefix rel_norm max | prefix rel_maxabs max | block rel_norm mean / max | block rel_maxabs max | block min_cos | readout-token rel_norm max | dP(present) max | d presence logit max | center KL max | center argmax agree | box d max | label KL max | vocab KL max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fla.exactness.fp32/causal.same_dtype_ref` | 1.51e-04 | 1.43e-04 | 1.02e-04 / 2.10e-04 | 1.51e-04 | 0.9999996 | 1.81e-04 | 2.49e-04 | 0.0036 | 6.07e-07 | 0.9938 | 0.0167 | 3.39e-07 | 4.40e-07 |
| `fla.exactness.fp32/block_bidir.same_dtype_ref` | 1.51e-04 | 1.41e-04 | 1.03e-04 / 2.15e-04 | 9.86e-05 | 0.9999996 | 1.47e-04 | 3.48e-04 | 0.0022 | 4.85e-07 | 0.9938 | 0.0119 | 3.05e-07 | 3.46e-07 |
| `fla.exactness.bf16/causal.same_dtype_ref` | 0.0065 | 0.0146 | 0.0058 / 0.0072 | 0.0110 | 0.9999301 | 0.0116 | 0.0154 | 0.0642 | 0.0017 | 0.8875 | 0.0383 | 6.10e-04 | 6.81e-04 |
| `fla.exactness.bf16/causal.fp32_ref` | 0.0180 | 0.0332 | 0.0068 / 0.0184 | 0.0078 | 0.9994161 | 0.0333 | 0.0149 | 0.2451 | 0.0102 | 0.8812 | 0.3312 | 0.0034 | 0.0047 |
| `fla.exactness.bf16/causal.hf_bf16_vs_hf_fp32` | 0.0180 | 0.0332 | 0.0071 / 0.0198 | 0.0097 | 0.9992851 | - | 0.0172 | 0.2770 | 0.0129 | 0.8750 | 0.3329 | 0.0041 | 0.0049 |
| `fla.exactness.bf16/block_bidir.same_dtype_ref` | 0.0065 | 0.0144 | 0.0058 / 0.0076 | 0.0115 | 0.9999283 | 0.0096 | 0.0098 | 0.1090 | 7.14e-04 | 0.8750 | 0.0438 | 5.69e-04 | 8.25e-04 |
| `fla.exactness.bf16/block_bidir.fp32_ref` | 0.0180 | 0.0332 | 0.0066 / 0.0189 | 0.0097 | 0.9991564 | 0.0411 | 0.0111 | 0.1708 | 0.0098 | 0.8875 | 0.2015 | 0.0077 | 0.0070 |
| `fla.exactness.bf16/block_bidir.hf_bf16_vs_hf_fp32` | 0.0179 | 0.0332 | 0.0069 / 0.0200 | 0.0098 | 0.9990519 | - | 0.0175 | 0.1583 | 0.0119 | 0.9187 | 0.2101 | 0.0098 | 0.0080 |
| `torch_gdn_fp32.exactness.fp32/causal.same_dtype_ref` | 6.71e-07 | 1.39e-06 | 5.17e-07 / 9.31e-07 | 1.10e-06 | 0.9999996 | 9.76e-07 | 1.19e-06 | 1.00e-05 | 4.25e-07 | 1.0000 | 1.43e-06 | 2.12e-07 | 3.75e-07 |
| `torch_gdn_fp32.exactness.fp32/block_bidir.same_dtype_ref` | 6.69e-07 | 1.22e-06 | 5.13e-07 / 9.93e-07 | 9.30e-07 | 0.9999996 | 1.01e-06 | 1.88e-06 | 1.14e-05 | 5.86e-07 | 1.0000 | 2.68e-06 | 2.55e-07 | 4.53e-07 |

`same_dtype_ref` = engine vs HF in the same dtype; `fp32_ref` = engine bf16 vs HF fp32; `hf_bf16_vs_hf_fp32` = HF's own bf16 noise floor (no engine involved). `causal`: engine attn_mask=causal vs HF's default forward. `block_bidir` (deployed mode): vs HF with a dict mask (prefix causal, block bidirectional on the full-attention layers; GDN layers untouched). Center argmax agreement < 1 in bf16 occurs on near-flat center distributions (center KL max ~1e-2, same as HF bf16 vs fp32).

## 2. Leak of block-diagonal packing (prefix + 8 blocks in one sequence) vs isolated (prefix + block k)

Variants: `blockmask` = 4D mask on full-attention layers, prefix causal, block k sees prefix + itself (causal), positions restart after the prefix for each block; `blockmask_bidir` = same with bidirectional blocks (ref: isolated bidirectional); `nomask` = plain causal over the packed sequence, positions restart per block; `nomask_seqpos` = plain concatenation. Reference runs use the same attention kernel path as the packed run (`ref` below). Block 0 is the control (nothing precedes it but the prefix).

Prefix check (packed prefix hidden states vs isolated prefix hidden states):

- `fla.leak.fp32/blockmask.prefix`: rel_norm max 5.93e-05, rel_maxabs max 4.31e-05, min_cos 0.9999994, max over all 8 isolated runs 5.93e-05 (ref `causal_dict`)
- `fla.leak.fp32/blockmask_bidir.prefix`: rel_norm max 5.93e-05, rel_maxabs max 4.31e-05, min_cos 0.9999994, max over all 8 isolated runs 5.93e-05 (ref `bidir_dict`)
- `fla.leak.fp32/nomask.prefix`: rel_norm max 5.94e-05, rel_maxabs max 4.31e-05, min_cos 0.9999995, max over all 8 isolated runs 5.94e-05 (ref `causal_default`)
- `fla.leak.fp32/nomask_seqpos.prefix`: rel_norm max 5.94e-05, rel_maxabs max 4.31e-05, min_cos 0.9999995, max over all 8 isolated runs 5.94e-05 (ref `causal_default`)
- `fla.leak.bf16/blockmask.prefix`: rel_norm max 0.0064, rel_maxabs max 0.0145, min_cos 0.9999096, max over all 8 isolated runs 0.0064 (ref `causal_dict`)
- `fla.leak.bf16/blockmask_bidir.prefix`: rel_norm max 0.0064, rel_maxabs max 0.0145, min_cos 0.9999096, max over all 8 isolated runs 0.0064 (ref `bidir_dict`)
- `fla.leak.bf16/nomask.prefix`: rel_norm max 0.0065, rel_maxabs max 0.0145, min_cos 0.9999020, max over all 8 isolated runs 0.0065 (ref `causal_default`)
- `fla.leak.bf16/nomask_seqpos.prefix`: rel_norm max 0.0065, rel_maxabs max 0.0145, min_cos 0.9999020, max over all 8 isolated runs 0.0065 (ref `causal_default`)
- `torch_gdn_fp32.leak.fp32/blockmask.prefix`: rel_norm max 6.24e-07, rel_maxabs max 1.28e-06, min_cos 0.9999995, max over all 8 isolated runs 6.74e-07 (ref `causal_dict`)
- `torch_gdn_fp32.leak.fp32/blockmask_bidir.prefix`: rel_norm max 6.24e-07, rel_maxabs max 1.28e-06, min_cos 0.9999995, max over all 8 isolated runs 6.74e-07 (ref `bidir_dict`)
- `torch_gdn_fp32.leak.fp32/nomask.prefix`: rel_norm max 6.24e-07, rel_maxabs max 1.37e-06, min_cos 0.9999995, max over all 8 isolated runs 6.77e-07 (ref `causal_default`)
- `torch_gdn_fp32.leak.fp32/nomask_seqpos.prefix`: rel_norm max 6.24e-07, rel_maxabs max 1.37e-06, min_cos 0.9999995, max over all 8 isolated runs 6.77e-07 (ref `causal_default`)

### `fla.leak.fp32/blockmask.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.54e-05 / 6.19e-05 | 3.37e-05 | 8.34e-08 / 3.58e-07 | -3.45e-09 | 1.0000 | 2.96e-05 | 1.37e-07 | 5.03e-08 | 1.0000 |
| 1 | 0.1638 / 0.2032 | 0.0437 | 0.0222 / 0.2329 | 0.0061 | 0.8000 | 0.0846 | 0.0237 | 0.0088 | 0.9500 |
| 2 | 0.1771 / 0.3103 | 0.0870 | 0.0287 / 0.2310 | 0.2322 | 0.5500 | 0.7739 | 1.0893 | 0.0470 | 0.8000 |
| 3 | 0.1683 / 0.2092 | 0.0769 | 0.0370 / 0.1620 | 0.0256 | 0.6000 | 0.2766 | 0.0722 | 0.0293 | 0.8000 |
| 4 | 0.1568 / 0.1897 | 0.0636 | 0.0263 / 0.2740 | 0.0154 | 0.7500 | 0.2963 | 0.0514 | 0.0177 | 0.9000 |
| 5 | 0.1618 / 0.2103 | 0.0639 | 0.0158 / 0.1016 | 0.0071 | 0.7500 | 0.0363 | 0.0422 | 0.0168 | 0.8500 |
| 6 | 0.1688 / 0.2557 | 0.0755 | 0.0201 / 0.1348 | 0.0846 | 0.8500 | 0.5331 | 0.4494 | 0.0251 | 0.8000 |
| 7 | 0.1560 / 0.1869 | 0.0614 | 0.0062 / 0.0326 | 0.0076 | 0.7500 | 0.1119 | 0.0471 | 0.0132 | 0.9000 |

### `fla.leak.fp32/blockmask_bidir.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.53e-05 / 6.14e-05 | 3.11e-05 | 1.13e-07 / 7.15e-07 | -4.08e-09 | 1.0000 | 2.88e-05 | 1.26e-07 | -5.43e-08 | 1.0000 |
| 1 | 0.1609 / 0.1898 | 0.0439 | 0.0270 / 0.2848 | 0.0072 | 0.8000 | 0.1422 | 0.0206 | 0.0093 | 0.9000 |
| 2 | 0.1708 / 0.2792 | 0.0786 | 0.0289 / 0.2609 | 0.1023 | 0.5000 | 0.8127 | 0.3408 | 0.0358 | 0.8500 |
| 3 | 0.1557 / 0.1882 | 0.0662 | 0.0221 / 0.1475 | 0.0212 | 0.6500 | 0.0942 | 0.1063 | 0.0240 | 0.8500 |
| 4 | 0.1581 / 0.1941 | 0.0604 | 0.0168 / 0.1738 | 0.0164 | 0.7000 | 0.3727 | 0.0269 | 0.0151 | 0.9000 |
| 5 | 0.1583 / 0.2026 | 0.0597 | 0.0132 / 0.1028 | 0.0066 | 0.7000 | 0.0640 | 0.0359 | 0.0153 | 0.7500 |
| 6 | 0.1626 / 0.2558 | 0.0663 | 0.0125 / 0.0898 | 0.0880 | 0.8000 | 0.5425 | 0.5191 | 0.0207 | 0.8000 |
| 7 | 0.1587 / 0.1993 | 0.0564 | 0.0072 / 0.0804 | 0.0068 | 0.8000 | 0.1409 | 0.0360 | 0.0108 | 0.9500 |

### `fla.leak.fp32/nomask.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.66e-05 / 7.93e-05 | 3.14e-05 | 8.34e-08 / 5.96e-07 | -3.42e-08 | 1.0000 | 2.50e-05 | 1.72e-07 | 1.91e-08 | 1.0000 |
| 1 | 0.1645 / 0.1990 | 0.0479 | 0.0203 / 0.2063 | 0.0066 | 0.7500 | 0.1420 | 0.0153 | 0.0102 | 0.8000 |
| 2 | 0.1750 / 0.2698 | 0.0863 | 0.0244 / 0.2682 | 0.1254 | 0.6500 | 0.7890 | 0.2350 | 0.0399 | 0.7500 |
| 3 | 0.1613 / 0.1919 | 0.0715 | 0.0182 / 0.1309 | 0.0178 | 0.7000 | 0.0366 | 0.0427 | 0.0231 | 0.8000 |
| 4 | 0.1632 / 0.2034 | 0.0675 | 0.0227 / 0.1717 | 0.0142 | 0.7500 | 0.2856 | 0.0383 | 0.0178 | 0.8000 |
| 5 | 0.1632 / 0.2497 | 0.0672 | 0.0135 / 0.0696 | 0.0081 | 0.7500 | 0.0307 | 0.0918 | 0.0216 | 0.7500 |
| 6 | 0.1624 / 0.2579 | 0.0709 | 0.0172 / 0.1461 | 0.1081 | 0.8000 | 0.5498 | 0.5805 | 0.0218 | 0.8000 |
| 7 | 0.1636 / 0.2021 | 0.0663 | 0.0044 / 0.0349 | 0.0170 | 0.8000 | 0.1117 | 0.0524 | 0.0143 | 0.9500 |

### `fla.leak.fp32/nomask_seqpos.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.66e-05 / 7.93e-05 | 3.14e-05 | 8.34e-08 / 5.96e-07 | -3.42e-08 | 1.0000 | 2.50e-05 | 1.72e-07 | 1.91e-08 | 1.0000 |
| 1 | 0.1687 / 0.2004 | 0.0533 | 0.0195 / 0.1587 | 0.0088 | 0.8000 | 0.1414 | 0.0179 | 0.0142 | 0.8500 |
| 2 | 0.1799 / 0.2802 | 0.0908 | 0.0270 / 0.2778 | 0.1076 | 0.5500 | 0.7914 | 0.2238 | 0.0516 | 0.7500 |
| 3 | 0.1618 / 0.2184 | 0.0701 | 0.0192 / 0.1047 | 0.0109 | 0.6000 | 0.0752 | 0.0749 | 0.0193 | 0.9000 |
| 4 | 0.1692 / 0.2221 | 0.0753 | 0.0220 / 0.1645 | 0.0260 | 0.7500 | 0.2061 | 0.0555 | 0.0243 | 0.8000 |
| 5 | 0.1681 / 0.2524 | 0.0697 | 0.0068 / 0.0356 | 0.0108 | 0.7500 | 0.0383 | 0.1005 | 0.0257 | 0.7500 |
| 6 | 0.1641 / 0.2378 | 0.0724 | 0.0196 / 0.1504 | 0.0547 | 0.7500 | 0.5217 | 0.3188 | 0.0204 | 0.8000 |
| 7 | 0.1725 / 0.2181 | 0.0753 | 0.0086 / 0.0685 | 0.0217 | 0.6500 | 0.0901 | 0.0532 | 0.0209 | 0.9000 |

### `fla.leak.bf16/blockmask.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0060 / 0.0073 | 0.0060 | 7.96e-06 / 3.48e-05 | 1.00e-04 | 0.9000 | 0.0294 | 1.06e-04 | 3.05e-04 | 1.0000 |
| 1 | 0.1640 / 0.2033 | 0.0438 | 0.0212 / 0.2252 | 0.0063 | 0.7500 | 0.1458 | 0.0255 | 0.0090 | 0.8000 |
| 2 | 0.1774 / 0.3083 | 0.0871 | 0.0301 / 0.2370 | 0.2253 | 0.4500 | 0.7751 | 1.0349 | 0.0461 | 0.8000 |
| 3 | 0.1680 / 0.2099 | 0.0757 | 0.0373 / 0.1588 | 0.0179 | 0.6500 | 0.0737 | 0.0731 | 0.0270 | 0.7500 |
| 4 | 0.1571 / 0.1905 | 0.0633 | 0.0269 / 0.2808 | 0.0157 | 0.7500 | 0.0705 | 0.0535 | 0.0176 | 0.8500 |
| 5 | 0.1624 / 0.2140 | 0.0643 | 0.0154 / 0.0936 | 0.0071 | 0.8000 | 0.0380 | 0.0394 | 0.0171 | 0.8500 |
| 6 | 0.1686 / 0.2546 | 0.0761 | 0.0199 / 0.1335 | 0.0906 | 0.7500 | 0.5153 | 0.4895 | 0.0257 | 0.9000 |
| 7 | 0.1563 / 0.1865 | 0.0624 | 0.0063 / 0.0344 | 0.0079 | 0.6500 | 0.1117 | 0.0491 | 0.0141 | 0.8500 |

### `fla.leak.bf16/blockmask_bidir.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0058 / 0.0067 | 0.0056 | 9.06e-06 / 3.87e-05 | 1.15e-04 | 0.9000 | 0.0294 | 1.79e-04 | 2.88e-04 | 1.0000 |
| 1 | 0.1608 / 0.1896 | 0.0443 | 0.0261 / 0.2688 | 0.0073 | 0.7500 | 0.1413 | 0.0233 | 0.0097 | 0.9000 |
| 2 | 0.1708 / 0.2722 | 0.0786 | 0.0303 / 0.2683 | 0.0937 | 0.5000 | 0.8015 | 0.2190 | 0.0347 | 0.8500 |
| 3 | 0.1557 / 0.1884 | 0.0661 | 0.0228 / 0.1708 | 0.0204 | 0.6500 | 0.0920 | 0.1059 | 0.0243 | 0.8000 |
| 4 | 0.1584 / 0.1943 | 0.0603 | 0.0170 / 0.1659 | 0.0176 | 0.8000 | 0.3262 | 0.0249 | 0.0145 | 0.8500 |
| 5 | 0.1589 / 0.2033 | 0.0606 | 0.0130 / 0.1040 | 0.0067 | 0.7000 | 0.0451 | 0.0360 | 0.0157 | 0.7000 |
| 6 | 0.1626 / 0.2541 | 0.0672 | 0.0128 / 0.0922 | 0.0884 | 0.7500 | 0.5452 | 0.5255 | 0.0211 | 0.8500 |
| 7 | 0.1591 / 0.1997 | 0.0571 | 0.0074 / 0.0795 | 0.0070 | 0.7000 | 0.1388 | 0.0369 | 0.0116 | 0.8500 |

### `fla.leak.bf16/nomask.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0059 / 0.0069 | 0.0054 | 7.67e-06 / 3.28e-05 | 1.04e-04 | 0.9500 | 0.0081 | 6.83e-05 | 3.13e-04 | 1.0000 |
| 1 | 0.1647 / 0.1996 | 0.0484 | 0.0199 / 0.2024 | 0.0066 | 0.7500 | 0.1431 | 0.0166 | 0.0104 | 0.8500 |
| 2 | 0.1749 / 0.2660 | 0.0863 | 0.0242 / 0.2647 | 0.1197 | 0.5000 | 0.7968 | 0.2245 | 0.0387 | 0.7500 |
| 3 | 0.1612 / 0.1890 | 0.0705 | 0.0184 / 0.1271 | 0.0121 | 0.6000 | 0.0906 | 0.0400 | 0.0216 | 0.8000 |
| 4 | 0.1636 / 0.2062 | 0.0673 | 0.0235 / 0.1722 | 0.0143 | 0.6500 | 0.0523 | 0.0364 | 0.0180 | 0.8000 |
| 5 | 0.1641 / 0.2534 | 0.0681 | 0.0136 / 0.0663 | 0.0083 | 0.7500 | 0.0315 | 0.0941 | 0.0221 | 0.7500 |
| 6 | 0.1623 / 0.2560 | 0.0714 | 0.0164 / 0.1331 | 0.1095 | 0.7500 | 0.5526 | 0.5783 | 0.0224 | 0.9000 |
| 7 | 0.1639 / 0.2017 | 0.0673 | 0.0044 / 0.0341 | 0.0170 | 0.7500 | 0.1137 | 0.0525 | 0.0152 | 0.8500 |

### `fla.leak.bf16/nomask_seqpos.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0059 / 0.0069 | 0.0054 | 7.67e-06 / 3.28e-05 | 1.04e-04 | 0.9500 | 0.0081 | 6.83e-05 | 3.13e-04 | 1.0000 |
| 1 | 0.1689 / 0.2009 | 0.0542 | 0.0192 / 0.1582 | 0.0089 | 0.7500 | 0.1425 | 0.0170 | 0.0152 | 0.8000 |
| 2 | 0.1797 / 0.2767 | 0.0910 | 0.0275 / 0.2783 | 0.0999 | 0.4500 | 0.8002 | 0.1844 | 0.0505 | 0.7500 |
| 3 | 0.1617 / 0.2190 | 0.0698 | 0.0202 / 0.1065 | 0.0081 | 0.5500 | 0.0765 | 0.0751 | 0.0186 | 0.8000 |
| 4 | 0.1698 / 0.2246 | 0.0749 | 0.0230 / 0.1711 | 0.0239 | 0.5500 | 0.0928 | 0.0552 | 0.0238 | 0.8000 |
| 5 | 0.1691 / 0.2577 | 0.0704 | 0.0074 / 0.0339 | 0.0111 | 0.7500 | 0.0382 | 0.1075 | 0.0263 | 0.7500 |
| 6 | 0.1642 / 0.2380 | 0.0732 | 0.0186 / 0.1547 | 0.0604 | 0.6500 | 0.5287 | 0.3474 | 0.0215 | 0.8500 |
| 7 | 0.1727 / 0.2177 | 0.0759 | 0.0085 / 0.0679 | 0.0218 | 0.7000 | 0.0896 | 0.0554 | 0.0219 | 0.7500 |

### `torch_gdn_fp32.leak.fp32/blockmask.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.48e-07 / 7.15e-07 | 4.96e-07 | 0.0000 / 0.0000 | 3.05e-09 | 1.0000 | 2.38e-07 | 2.30e-07 | 7.35e-09 | 1.0000 |
| 1 | 0.1639 / 0.2033 | 0.0437 | 0.0222 / 0.2328 | 0.0061 | 0.8000 | 0.0847 | 0.0237 | 0.0088 | 0.9000 |
| 2 | 0.1772 / 0.3104 | 0.0871 | 0.0289 / 0.2322 | 0.2323 | 0.5500 | 0.7739 | 1.0898 | 0.0471 | 0.8000 |
| 3 | 0.1684 / 0.2093 | 0.0770 | 0.0371 / 0.1624 | 0.0256 | 0.6000 | 0.2768 | 0.0723 | 0.0294 | 0.8000 |
| 4 | 0.1568 / 0.1898 | 0.0636 | 0.0263 / 0.2740 | 0.0153 | 0.7500 | 0.2958 | 0.0513 | 0.0177 | 0.9000 |
| 5 | 0.1618 / 0.2104 | 0.0639 | 0.0158 / 0.1020 | 0.0071 | 0.7500 | 0.0362 | 0.0423 | 0.0168 | 0.8500 |
| 6 | 0.1689 / 0.2559 | 0.0756 | 0.0201 / 0.1350 | 0.0849 | 0.8500 | 0.5332 | 0.4519 | 0.0251 | 0.8000 |
| 7 | 0.1561 / 0.1869 | 0.0615 | 0.0062 / 0.0327 | 0.0076 | 0.7500 | 0.1120 | 0.0472 | 0.0133 | 0.9000 |

### `torch_gdn_fp32.leak.fp32/blockmask_bidir.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.42e-07 / 6.58e-07 | 4.86e-07 | 0.0000 / 0.0000 | -7.18e-08 | 1.0000 | 2.38e-07 | 9.64e-08 | 6.76e-08 | 1.0000 |
| 1 | 0.1609 / 0.1899 | 0.0439 | 0.0270 / 0.2846 | 0.0072 | 0.8000 | 0.1420 | 0.0206 | 0.0093 | 0.9000 |
| 2 | 0.1709 / 0.2793 | 0.0787 | 0.0290 / 0.2617 | 0.1024 | 0.4500 | 0.8128 | 0.3417 | 0.0358 | 0.8500 |
| 3 | 0.1558 / 0.1883 | 0.0662 | 0.0222 / 0.1484 | 0.0213 | 0.6000 | 0.0941 | 0.1061 | 0.0240 | 0.8500 |
| 4 | 0.1582 / 0.1942 | 0.0605 | 0.0168 / 0.1739 | 0.0164 | 0.7000 | 0.3737 | 0.0269 | 0.0152 | 0.9000 |
| 5 | 0.1584 / 0.2027 | 0.0598 | 0.0132 / 0.1032 | 0.0066 | 0.7000 | 0.0640 | 0.0360 | 0.0153 | 0.7500 |
| 6 | 0.1626 / 0.2559 | 0.0663 | 0.0126 / 0.0902 | 0.0882 | 0.8000 | 0.5425 | 0.5202 | 0.0207 | 0.8000 |
| 7 | 0.1588 / 0.1994 | 0.0565 | 0.0073 / 0.0806 | 0.0068 | 0.8000 | 0.1412 | 0.0359 | 0.0109 | 0.9500 |

### `torch_gdn_fp32.leak.fp32/nomask.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.66e-07 / 8.18e-07 | 5.23e-07 | 0.0000 / 0.0000 | -2.70e-08 | 1.0000 | 3.87e-07 | 1.39e-07 | -2.83e-08 | 1.0000 |
| 1 | 0.1646 / 0.1990 | 0.0479 | 0.0203 / 0.2064 | 0.0066 | 0.7500 | 0.1418 | 0.0153 | 0.0102 | 0.8500 |
| 2 | 0.1750 / 0.2698 | 0.0864 | 0.0245 / 0.2690 | 0.1253 | 0.6500 | 0.7890 | 0.2336 | 0.0399 | 0.7500 |
| 3 | 0.1614 / 0.1921 | 0.0716 | 0.0182 / 0.1314 | 0.0179 | 0.7500 | 0.0373 | 0.0429 | 0.0232 | 0.8000 |
| 4 | 0.1632 / 0.2036 | 0.0676 | 0.0227 / 0.1717 | 0.0142 | 0.7500 | 0.2852 | 0.0385 | 0.0179 | 0.8000 |
| 5 | 0.1633 / 0.2497 | 0.0673 | 0.0135 / 0.0697 | 0.0081 | 0.7500 | 0.0308 | 0.0920 | 0.0217 | 0.7500 |
| 6 | 0.1625 / 0.2580 | 0.0709 | 0.0173 / 0.1466 | 0.1082 | 0.8000 | 0.5498 | 0.5814 | 0.0219 | 0.8000 |
| 7 | 0.1636 / 0.2021 | 0.0664 | 0.0044 / 0.0350 | 0.0170 | 0.8000 | 0.1119 | 0.0524 | 0.0143 | 0.9500 |

### `torch_gdn_fp32.leak.fp32/nomask_seqpos.per_block.<k>`

| k | block hidden rel_norm mean / max | readout-token rel_norm mean | dP(present) mean / max | center KL mean | center argmax agree | box d max | label KL max | vocab KL mean | vocab top1 agree |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.66e-07 / 8.18e-07 | 5.23e-07 | 0.0000 / 0.0000 | -2.70e-08 | 1.0000 | 3.87e-07 | 1.39e-07 | -2.83e-08 | 1.0000 |
| 1 | 0.1687 / 0.2004 | 0.0534 | 0.0196 / 0.1589 | 0.0089 | 0.8000 | 0.1412 | 0.0179 | 0.0142 | 0.8000 |
| 2 | 0.1799 / 0.2803 | 0.0909 | 0.0271 / 0.2786 | 0.1077 | 0.5500 | 0.7914 | 0.2241 | 0.0516 | 0.7500 |
| 3 | 0.1619 / 0.2185 | 0.0702 | 0.0193 / 0.1053 | 0.0110 | 0.6500 | 0.0754 | 0.0750 | 0.0193 | 0.9000 |
| 4 | 0.1692 / 0.2222 | 0.0753 | 0.0220 / 0.1644 | 0.0260 | 0.7500 | 0.2057 | 0.0557 | 0.0244 | 0.8000 |
| 5 | 0.1682 / 0.2524 | 0.0697 | 0.0068 / 0.0355 | 0.0108 | 0.7500 | 0.0384 | 0.1008 | 0.0257 | 0.7500 |
| 6 | 0.1641 / 0.2379 | 0.0724 | 0.0196 / 0.1501 | 0.0548 | 0.7500 | 0.5216 | 0.3192 | 0.0205 | 0.8000 |
| 7 | 0.1725 / 0.2182 | 0.0754 | 0.0086 / 0.0684 | 0.0218 | 0.7000 | 0.0903 | 0.0532 | 0.0210 | 0.9000 |

## 3. FlexAttention

- `fla.flex`: did not run. ValueError: Qwen3_5ForConditionalGeneration does not support an attention implementation through torch's flex_attention (transformers 5.23.0). No flex numbers are reported.

## 4. Relation to the 27B test (`outputs/27b/bench/leak.json`)

There the packed-vs-isolated prefix differed by 0.38 rel_norm, which cannot be leakage (the prefix is causal and comes first). Here the same check gives float noise (section 2), so the 27B prefix gap was a harness artifact, and the per-block numbers in this file supersede it for the 4B claim. Negative KL values of order 1e-8 are float rounding of a zero KL.

