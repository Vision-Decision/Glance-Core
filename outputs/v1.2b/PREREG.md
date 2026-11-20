# Glance v1.2b 预注册（多框 head，建在 v1.2 之上）

写于 2026-11-18，v1.2b 开训之前。

- 只评最终检查点，评一次。
- 解码阈值只在 calib_mb 上选，不碰任何评测集。

## 训练方案（沿用 multi-box pilot）

**第一阶段：只训多框 heads**，450 步。

**第二阶段：联合训练**，1200 步：

- 学习率：LoRA 2e-5，heads 5e-5，多框 heads 5e-5；
- `multi_backbone_grad=0`：多框 heads 的梯度不回传到 backbone；
- `center_multi=False`，多框 heads 单独做梯度裁剪。

**解码**：默认用 count 解码，取 P(match) 最高的 n 个框，n 由 count head 给出。

## 验收标准（对照 v1.2 同一套设置；pilot 指 jointnb2）

| # | 条目 | 标准 |
|---|---|---|
| M1 | gRefCOCO val 多目标题的集合 F1（count 解码） | ≥ 70（v1.1 单框 51.0，pilot 74.4） |
| M2 | cocoall_val 多实例题的集合 F1 | 高于 v1.2 单框 |
| M3 | Glance-Data 诊断集 neg_count 的 N-acc | ≥ v1.2 − 3（pilot 跌到了 5） |
| M4 | RefCOCO/+/g 8 个 split 平均 Acc@0.5 | ≥ v1.2 − 0.5（pilot 降了 1.2） |
| M5 | gRefCOCO val N-acc | ≥ v1.2 − 1 |
| M6 | gRefCOCO val 多目标集合 F1，按 Glance-X 的定义计算 | ≥ 77.3（Glance-X i2t 自报的数字，我们没有复核） |

M6 的定义：

- 只算有 2 个及以上目标的表达式；
- 预测集合不经过存在门控；
- 按 IoU ≥ 0.5 贪心做一对一匹配，每个表达式算 F1 = 2TP / (|P| + |G|)，再对所有表达式取平均；
- 解码阈值在 calib_mb 上选。

计算脚本是 `scripts/v12b/multi_f1.py`，定义出自 `reports/Glance-X 评估.md` §2。

## 多框 head 对决（2026-11-19 追加，开训前）

**对比的两个 head**：都建在同一个 v1.2 骨干上，用同一份数据（`prep_v12b`）、同样的两阶段训练和步数（450 + 1200）。

- **A**：v12b 的方案，热图 + 集合匹配 + 计数头。
- **B**：Glance-X 的逐 patch objectness 头，从 Apache-2.0 代码移植而来（`multi_head=objectness`）。损失用 sigmoid focal，正样本取框中心 50% 的区域；候选为 top-k 个 patch，P(match) 直接取 objectness；初始化时偏置设为 −4。

**胜负规则**：

1. 先过约束：M4（8 个 split 的平均 Acc@0.5 ≥ v1.2 − 1）和 M3（neg_count N-acc ≥ v1.2 − 3）。
2. 在过了约束的 head 里，M6 高者胜。
3. 两个都没过约束时，取 M6 更高的那个，并把它违反了哪条约束如实报告。

胜者进入 v1.3。
