# Glance v1.2 预注册验收标准

写于 2026-11-18，在 v1.2 训练开始之前提交。训练结束后只评一次最终检查点（`ckpt_final.pt`），逐条报告通过或不通过。

规则：

- 不在任何评测集上挑检查点、调阈值或调配比。
- 温度只在训练分布的留出集上拟合（`calib_natural`、`calib_text`），与 v1.1 相同。
- 有不通过的条目时，提出修复方案，下一版再验证，不回头调这一版。

## 模型与对照

- **v1.2**：从 v1.1 的 `ckpt_final.pt` 继续训练，LoRA 和 heads 结构不变。
  - 配置见 `configs/v1.2_4b.json`。
  - 一致性损失 `w_cons=0.5`，定义见 `dv/model/consistency.py`。
- **对照**：v1 和 v1.1 的已有结果。
  - `outputs/ground_v2/diag/`
  - HF `Vision-Decision/glance-train-ckpts-legacy` 下的 `robust_eval/scores_all.json`、`v1.1/eval_results*.json`
  - S2 sprint 的官方协议数字

## 验收标准

阈值 0.5 均指原始 P(present)。各数据集的评测方式写在"指标"一栏，v1.1 的值作为参考。

### A. 描述偏移（`Vision-Decision/glance-robust-ref`，`scripts/robust/score.py`，不加温度）

| # | 条目 | 指标 | v1.1 | 标准 |
|---|---|---|---|---|
| A1 | 纯功能说法（func / function_only） | 错判为不存在的比例 | 36.8% | ≤ 10% |
| A2 | 纯类别泛称（vague / generic，正例） | 错判为不存在的比例 | 44.1% | ≤ 10% |
| A3 | 关系说法（vague / relational，正例） | 错判为不存在的比例 | 50.7% | ≤ 10% |
| A4 | 关系说法（vague / relational） | 门控 Acc@0.5 | 13.6 | ≥ 40 |

### B. Glance-Data 诊断集（300 张 refcoco_val 图，加 `diagnostic_pos_sole`，`scripts/ground_v2/eval_diag.py`，取 `/all` 即中英合并）

| # | 条目 | 指标 | v1 | v1.1 | 标准 |
|---|---|---|---|---|---|
| B1 | neg_count | N-acc | 66.7 | 46.9 | ≥ 65 |
| B2 | pos_pos_sole（单实例上的方位词和最高级） | 接受率（P ≥ 0.5） | 33.9 | 33.8 | ≥ 70 |
| B3 | cat_only（裸类别名） | 门控 Acc@0.5 | 50.0 | 44.1 | ≥ 60 |
| B4 | hneg_attribute | N-acc | 67.3 | 68.4 | ≥ 68.4（不低于 v1.1） |
| B5 | hneg_relation | N-acc | 35.3 | 30.0 | ≥ 30.0（不低于 v1.1） |

### C. 保住 v1.1 的成绩（`eval.py`，设置与 `scripts/v1/post_train_v11.sh` 相同）

| # | 条目 | 指标 | v1.1 | 标准 |
|---|---|---|---|---|
| C1 | RefCOCO val | Acc@0.5（不门控） | 94.9 | ≥ 94 |
| C2 | RefCOCO / + / g 共 8 个 split | Acc@0.5 平均 | 92.8 | ≥ 92.5 |
| C3 | gRefCOCO val（merged） | N-acc，阈值 0.5 | 86.0 | ≥ 86 |
| C4 | gRefCOCO val（merged） | presence ECE（用 calib_natural 拟合的温度校准后） | 0.036 | ≤ 0.03 |
| C5 | VenusBench-GD，`summarize_gui` 口径 | refusal_acc | 82.2 | ≥ 81 |
| C6 | VenusBench-GD，`summarize_gui` 口径 | grounding（click_acc） | 55.2 | ≥ 53 |
| C7 | ScreenSpot-v2 | click_acc | 89.2 | ≥ 88 |
| C8 | OpenDecisions | capability（raw） | 67.8 | ≥ 66 |

- C2 的 8 个 split：refcoco val / testA / testB，refcoco+ val / testA / testB，refcocog val / test。
- C5–C7 的 GUI 设置：`--gui_max_vis 8192 --gui_upscale 1.0`，单次前向。
- C8 同时报告校准后的 capability。

### D. 官方协议（S2 sprint 口径，协调者 2026-11-18 追加）

| # | 条目 | 指标 | v1 | v1.1 | 标准 |
|---|---|---|---|---|---|
| D1 | VenusBench-GD 官方 UI-Venus `eval.py`，忠实映射 | Overall（原生分辨率，G4） | 40.1 | 40.7（2048 GUI tokens） | ≥ 45 |
| D2 | 同上 | Refusal | 81.8 | — | ≥ 80 |
| D3 | 同上 | 存在目标的接受率（P ≥ 0.5） | 53.5% | — | ≥ 70% |
| D4 | GREC 官方（gRefCOCO `refexp.py` 移植，阈值 0.7），val | N-acc | 91.4 | 89.6 | ≥ 90 |
| D5 | 同上 | Pr | 71.6 | 72.2 | ≥ 72 |

- D1 的忠实映射：Glance 判为不存在时不给点击。v1 的分项为 Basic 42.4 / Refusal 81.8 / Advanced 37.1。
- 早先的 47.3 / 52.3 在拒答时仍然给了点击，不使用。

## 只报告、不设门槛

- ScreenSpot-Pro：v1.1 为 46.8。单次前向，另报 crop-zoom 的结果。
- robust-ref 的 zh、para、function_attr、generic_attr、absent 各子集。
- 诊断集的其余题型。
- 校准：原始 ECE，另报 S2 的免训练校准结果，即 ĝ 上的 Platt，加上无监督 EM 先验校正，再乘以 Platt 校准的 κ̂（`scripts/sprint_abs/`）。
- 延迟。

## 训练数据的事先声明（防止事后解释）

- **Glance-Data**：v1.2 用了 Glance-Data（`single` 视图，24,406 张 train2014 图）。
  - B 组诊断集来自 Glance-Data 的 300 张留出图（refcoco_val）。题型分布与训练同源，B 组因此是**同分布**检验，不是迁移检验。
  - A 组 robust-ref 的说法由另一条流水线（DeepSeek）独立生成，图也与训练无交集，是迁移检验。
- **新生成的无类别词数据**（`scripts/v12/catfree_*`）：
  - 图像不与任何评测集或 Glance-Data 重叠，`build_mix.py` 里有断言。
  - 说法由 Qwen3.8-27B 生成，不是 robust-ref 用的 DeepSeek。
  - 类别集合与 robust-ref 有重叠（都是 COCO 80 类），说法风格可能相近，报告里会注明。

## 训练前的设置变更（2026-11-18，仍在训练开始前；验收标准没有改动）

- **一致性损失**：改用 S3 sprint 已验证的实现（`train.py consistency_loss`，字段为 `grp` / `anchor`），并且只作用于存在型分组（`cons_present_only`）。原因：S3 发现把缺席成员拉向锚点会让 N-acc 下降约 5 点。
- **每类无类别词正例都配同类负例**：负例按功能 / 泛称 / 关系三类，与正例数量成比例抽样（S3 的教训 2）。
- **box 题文本上限**：`max_q_tokens` 从 40 提到 96。原因：S1 发现 VenusBench 的推理类、功能类长指令在训练时被截断。
- **GUI 评测**：使用 main 上修正后的 `eval.py`（e047479，GUI 记录按真实 prefix 长度分批），在原生分辨率下评测（8192 tokens，不放大）。
- **基础回放量**：取 v1.1 的约 75%（65 万题）。原因：从 v1.1 起训只需要回放，这样也能把训练控制在约 4k 步。
