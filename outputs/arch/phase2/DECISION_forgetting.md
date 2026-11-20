# Phase 2 decision note: recovering ScreenSpot-Pro by changing where and how the LoRA learns

Written before training on 2026-11-19. Final numbers are added in the results section below once they are in.

## Why
- Exp 1 (α sweep, zero training): scaling the v1.1 LoRA delta to α = 0.7 lifts ScreenSpot-Pro native from 50.1–50.7 to **55.3**, with no other change. RefCOCO stays where it was: testA 96.1 vs 96.2, testB 92.4 vs 92.9, val 94.7 vs 94.9. The cost lands on the presence gate: gRefCOCO N-acc at p ≥ 0.5 falls from 0.860 to 0.820. So at least ~5 of the 9 points between Glance and the base model (59.9 generative) come from the LoRA delta overwriting base GUI ability (forgetting), not from the patch-grid readout. The rest is pixels: exp 2 zoom adds +7 to +10 on top.
- Literature (report §10): forgetting depends on *where* the update sits. Attention-only placement gave held-out −0.6 vs −23 for full fine-tuning, and gate/up-only gave −2.1. Self-generated replay (SDFT) and KL-to-base cut it further.

## What (each run is 1,200 steps from base Qwen3.5-4B on the v1.1 replay mixture `train_base.jsonl`, v1.1 hyper-parameters, one G4 each)
| Run | Change vs v1.1 recipe | Tests |
|---|---|---|
| P0 control | none (LoRA r64 on every LM linear) | short-schedule reference; v1.1 itself at 3,857 steps is the long reference |
| P1 attn-only | LoRA only on full-attention q/k/v/o and the GDN token mixers (in_proj_qkv, in_proj_z, out_proj); no MLP | placement |
| P2 mlp-gate/up | LoRA only on MLP gate_proj/up_proj (down frozen) | placement |
| P3 replay | P0 + LM loss on base-generated GUI click answers (w_lm 0.2, kept only where the base click is correct) | SDFT-style replay = KL-to-base on GUI |

## Pre-registered success criterion (decided before any result)
- The variant must beat P0 on ScreenSpot-Pro native by ≥ 2 points, with a paired bootstrap 95% CI above 0.
- It must stay within 1 point of P0 on RefCOCO testA/testB Acc@0.5 and within 2 points of P0 on gRefCOCO N-acc.
- Stretch goal: SS-Pro ≥ 57 at full length (report item 9).

## What would change the plan
- If no placement beats P0, forgetting is not placement-sensitive in this hybrid model. The next step is then the task-gated adapter (item 15), which leaves the GUI path bit-identical to the base, plus α-scaling at inference with re-fitted presence temperatures.
- If replay wins, it is cheap to fold into v1.3 (≈ +5% step time).

## 结果（P0–P3）
`results_placement.json` 中的配对差值，都是相对 P0（全线性层 LoRA，1,200 步，同一份 279,805 条记录的混合数据）：

| 运行 | SS-Pro 原生 | Δ vs P0 [95% CI] | RefCOCO testA | testB | gRef N-acc |
|---|---|---|---|---|---|
| P0 对照 | 41.3 | — | 92.1 | 86.7 | 0.843 |
| P1 只放注意力 / GDN 混合层 | 43.0 | +1.7 [−0.4, +3.8] | 91.6（−0.5） | 86.5（−0.2） | 0.832 |
| P2 只放 MLP gate/up | 37.8 | **−3.5 [−5.7, −1.2]** | 91.7 | 87.2 | 0.834 |
| P3 底座自生成 GUI 回放 | 40.3 | −1.0 [−3.5, +1.4] | 92.1 | 86.8 | 0.836 |

- **都没有通过预注册标准**（要求 ≥ 2 点且 CI 在 0 以上）。P1 方向是对的，但不显著；P2 显著更差；P3 没有效果。
- 1,200 步的短训练本身在 SS-Pro 上就比 v1.1（3,857 步，49.8）低很多，RefCOCO 也低 3–5 个点。所以这些短训对照只能说明"位置 / 回放"的相对作用，绝对水平不能和 v1.1 比。
- 对照实验 1：推理时把 α 设为 0.7，零成本得到 +5.6 [+4.0, +7.1]，COCO 不受影响。训练时的这几种干预都没有达到这个量级。
- **结论**：在这个混合模型和数据上，遗忘对 LoRA 放在哪里不敏感。推理时按输入类型选 α（GUI 用 0.5–0.7，自然图用 1.0）是目前收益最大的手段，也就是第 15 项"任务门控"的免训练实现。下一步不再做位置扫描，改为按任务门控的训练版本：GUI 样本在训练时也用小 α 前向，或使用独立的 GUI adapter。这需要先和 v1.3 的整体规划对齐，属于方向性决策。
