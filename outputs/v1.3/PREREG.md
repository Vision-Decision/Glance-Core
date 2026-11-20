# Glance v1.3 预注册

写于 2026-11-19，早于 v1.3 开训。

**评测规则**

- 只评最终检查点，且只评一次。
- 温度和解码阈值只在 calib 集上选。
- 不在任何评测集上挑检查点或调配比。
- 有不通过的条目就提出修复方案，下一版再验证，不回头调这一版。

## 模型

**数据**

- 用 `scripts/v13/build_mix_v13.py` 构建，按 `outputs/v1.2/regression_diagnosis.md` 的结论重新配比。
- 每一处改动及其原因写在该脚本的头部注释里。

**初始化**

| 情况 | 初始化来源 | 训练方式 |
|---|---|---|
| v12b 对决的胜者达到 M6（多目标 F1 ≥ 77.3） | 该胜者的联合训练检查点（v1.2 骨干加多框 head） | 多框 head 和单框 head 一起训练 |
| 胜者未达到 M6 | v1.2 的 `ckpt_final` | 只训单框 |

无论哪种情况，都沿用 v1.2 的这些设置：

- 学习率：LoRA 1e-4，heads 3e-4；
- 一致性损失：只作用于存在类分组，权重 0.5；
- `max_q_tokens` = 96。

**反遗忘设置**

- 如果 arch-exp 分支在开训前给出了胜出的反遗忘设置（LoRA α 缩放、LoRA 放置位置、对 base 的 KL），就采用，并在下文补记。
- 如果没有给出，保持 v1.2 的设置。

## 验收标准：沿用 v1.2 的全部门槛

A、B、C、D 四组的条目、指标定义和阈值与 `outputs/v1.2/PREREG.md` 完全相同，结果由 `scripts/v12/report.py` 和 S2 流水线生成。

- **A 组**：robust-ref 中，功能、泛称、关系三类说法把"有"判成"没有"的比例 ≤ 10%；关系说法的 Acc ≥ 40。
- **B 组**：Glance-Data 诊断集。
  - neg_count ≥ 65；
  - pos_pos_sole 的接受率 ≥ 70；
  - cat_only ≥ 60；
  - hneg_attribute 和 hneg_relation 不低于 v1.1 的 68.4 和 30.0。
- **C 组**：
  - RefCOCO val ≥ 94；8 个 split 平均 ≥ 92.5；
  - gRefCOCO N-acc ≥ 86，ECE ≤ 0.03；
  - Venus 拒答率 ≥ 81，定位 ≥ 53；
  - ScreenSpot-v2 ≥ 88；
  - OpenDecisions ≥ 66。
- **D 组**：
  - Venus 官方 faithful 口径：Overall ≥ 45，Refusal ≥ 80，有目标时的接受率 ≥ 70%；
  - GREC 官方 val：N-acc ≥ 90，Pr ≥ 72。

## 另外报告的项（不设门槛）

**文本先验探针**（arch-exp 的 exp5）

- 用一个只看问句、不看图的分类器，在 v1.3 训练混合上测 AUROC，并在诊断集上测同一指标。
- B 组每条结果旁都并列这个只看文本的基线。B 组模板化的负例只有在只看文本的基线接近 0.5 时，才能算作"会看图"。

**v1.2 退步项**

- robust-ref 的 zh、para 两个子集的 N-acc，以及 relational 子集的 N-acc。
- 计数的成对测试：同样的"two X"说法，放在 X 恰好有两个的图上，看接受率。

**如果带多框 head**

- v1.2b 的 M1–M6，口径同 `outputs/v1.2b/PREREG.md`。

**其他**

- ScreenSpot-Pro 单次推理的结果和 crop-zoom 结果（crop-zoom 已修复，见 commit "eval: predict_v1 returns rows in input order"）。
- 校准：原始 ECE 和 S2 校准后的 ECE 都报。
- 延迟。

## 补记（2026-11-19，训练开始前）

- **v1.3 只训单框**（`configs/v1.3_4b.json` = `v1.3_4b_single.json`），在单独一台 G4 上跑（`scripts/v13/vm_v13_standalone.sh`），不再等多框 head 的对比。多框 head 之后作为 v1.3b 加在 v1.3 上，用对比的胜者。
- **防遗忘设置没有加入。** arch-exp 第二阶段（LoRA 放置位置 / 自生成回放，`outputs/arch/phase2/DECISION_forgetting.md`）在开训时还没有训练结果；唯一已有的 α 缩放是推理期做法，不改训练配方。v1.3 评测时会把 α=0.7 作为推理期变体另外报一次，不算入通过与否的判定。
- **中文翻译质量**：往返过滤后，Opus 抽检 500 条，残余错误率 2.8%（低于约定的 5% 门槛）。
- **多框 head 对比改为公平重跑**：v1.2b 热图 head 在联合训练阶段发散（heat 损失 1.3 → 14），所以热图臂用修正后的设置（热图单独 LR 1e-5、单独梯度裁剪 0.5、focal 损失按有效格数归一）在相同数据和步数下重跑一次，胜者按重跑结果判定（`outputs/v1.2b/PREREG.md` 的规则不变）。两臂都记录选框统计（预测框数、计数完全正确率、选中框的精确率 / 召回率、候选集合的 oracle 召回）。
