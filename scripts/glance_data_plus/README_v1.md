---
license: other
language:
- en
- zh
pretty_name: Glance-Data-Plus
tags:
- visual-grounding
- referring-expression-comprehension
- coco
- hard-negatives
- train-only
---

# Glance-Data-Plus（私有，仅用于训练）

Glance-Data-Plus 是 [Vision-Decision/Glance-Data](https://huggingface.co/datasets/Vision-Decision/Glance-Data) 的训练端补充数据，由 Glance 组整理。它与 Glance-Data 结构相同，可以直接拼接；唯一多出的是行级 `source` 列。

**本数据集只有训练集，不包含任何评测数据。** 评测基准单独冻结，不在这里。所有图像都来自 COCO train2014，并且排除了以下评测集用到的全部图像：

- RefCOCO / RefCOCO+ / RefCOCOg 的 val 和 test 划分
- gRefCOCO 的 val / testA / testB
- COCO val2017
- 歧义探针图像
- glance-robust-ref
- Glance-Data 诊断集

排除规则写在 `build_plus.py` 和 `scripts/v12/eval_ids.py` 中，构建时会做断言检查。

## 数据格式

每一行对应"一张图 × 一个来源"，字段如下：

| 字段 | 说明 |
|---|---|
| `image_id` | COCO 图像 id |
| `file` | 文件名 |
| `width`、`height` | 图像宽高（像素） |
| `questions` | 问题列表，每个问题是 `{text, lang, kind, src, present, n_targets, boxes}`，与 Glance-Data 相同 |
| `source` | 本行的来源，取值见下表 |

`boxes` 采用 COCO 原图像素坐标，格式为 xyxy。

## 各类别数量

正负比一栏中，"—"表示该类别只有一种极性，配对对象注在后面。

| source | kind | 题数 | en | zh | 正例 | 负例 | 正负比 |
|---|---|---:|---:|---:|---:|---:|---|
| ground_v2_gap_rule | pos_pos_sole | 63,190 | 31,625 | 31,565 | 63,190 | 0 | 与 Glance-Data 的 neg_position（63,198）1:1 |
| v1.2_catfree | function_catfree | 63,339 | 46,014 | 17,325 | 63,339 | 0 | 对应 neg_function_absent：4.4:1 |
| v1.2_catfree | generic | 53,751 | 38,725 | 15,026 | 53,751 | 0 | 对应全部泛称负例（20,082）：2.7:1；v1.3 训练按 1.5:1 下采样 |
| v1.2_catfree | relation_generic | 24,158 | 18,032 | 6,126 | 24,158 | 0 | 对应全部关系负例（19,071）：1.27:1；v1.3 训练按 1:1 |
| v1.2_catfree | anchor_category | 30,045 | 30,045 | 0 | 30,045 | 0 | 一致性锚点 |
| v1.2_catfree | neg_function_absent | 14,468 | 10,863 | 3,605 | 0 | 14,468 | — |
| v1.2_catfree | neg_generic_absent | 15,940 | 10,999 | 4,941 | 0 | 15,940 | — |
| v1.2_catfree | neg_relation_generic | 15,969 | 12,930 | 3,039 | 0 | 15,969 | — |
| v1.3_negatives | neg_generic_absent | 4,142 | 2,756 | 1,386 | 0 | 4,142 | — |
| v1.3_negatives | neg_position_absent_class | 4,490 | 3,483 | 1,007 | 0 | 4,490 | 与 pos_pos_sole 配对 |
| v1.3_negatives | neg_relation_absent_class | 2,770 | 2,228 | 542 | 0 | 2,770 | — |
| v1.3_negatives | neg_relation_thing | 332 | 226 | 106 | 0 | 332 | — |
| v1.3_count_pairs | count_exact | 24,906 | 14,805 | 10,101 | 24,906 | 0 | 与 Glance-Data 的 neg_count（149,931）构成最小对 |
| v1.3_refcoco_zh | refcoco_zh | 66,441 | 0 | 66,441 | 66,441 | 0 | 英文原句见清单 |
| manifest | refcoco_replay（只有 id） | 360,968 | | | | | v1 自然图像池，RefCOCO / + / g 训练集表达式加困难负例 |

**合计**：123,533 行，46,525 张图，题目约 41 万条（不含 manifest）。


## 各类别的构建方式

### 生成方式说明

答案（框、计数、某个类别是否在图中）都由规则从 COCO 标注计算，LLM 只负责写句子。

LLM 写出的句子要经过两道过滤才能保留：

1. **盲验证**：同一个模型在不知道目标的情况下看带编号的图，重新指认这句话指的是哪个物体。正例必须指回目标，负例必须回答"图中没有"。
2. **规则过滤**：去掉泄漏了类别名或同义词的句子（`scripts/robust/cat_names.py` 和 `scripts/ground_v2/lexicon.py`）；同时检查语言、长度，以及句子是否提到了图上画的编号或框。

### pos_pos_sole（来源 `ground_v2_gap_rule`）

- **内容**：图中某个类别只有唯一一个实例，问题用方位词或最高级指代它，例如"the leftmost cat"、"最左边的猫"。
- **生成方式**：纯规则，不用 LLM。
- **作用**：与 Glance-Data 的 `neg_position` 按 1:1 配平。

### v1.2 无类别词数据（来源 `v1.2_catfree`）

问题里不出现目标的类别名，有以下几种：

| 类别 | 说明 | 例子 |
|---|---|---|
| 纯功能说法 `function_catfree` | 按用途指代 | "something to drink from" |
| 泛称说法 `generic` | 用上位类指代 | "那个家具" |
| 关系说法 `relation_generic` | 泛称 + 相对位置 | "冰箱左边那只动物" |
| 对应的负例 | 图中没有符合描述的物体 | `neg_function_absent`、`neg_generic_absent`、`neg_relation_generic` |
| 一致性锚点 `anchor_category` | 规则生成的类别名短语 | "the cup" |

- **生成方式**：句子由 Qwen3.8-27B-FP8 在 Colab G4 上用 vLLM 自托管生成，盲验证使用同一模型。
- **人工抽检**：56 条中 53 条正确。

### v1.3 负例（来源 `v1.3_negatives`）

| 类别 | 内容 | 例子 |
|---|---|---|
| `neg_relation_absent_class` | 类别不存在 + 关系 | "the animal left of the bench"，图中没有动物 |
| `neg_relation_thing` | "东西 / thing" + 关系；规则确认该侧没有任何标注物体 | — |
| `neg_position_absent_class` | 方位词 + 图中没有的类别 | "suitcase on the left side" |
| `neg_generic_absent` | 泛称 + 图中没有的上位类 | — |

- **生成方式**：Qwen3.8-27B-FP8 在 A100 上生成，并做盲验证。
- **人工抽检**：32 条中 29 条正确。
- **为什么加这些**：针对 `outputs/v1.2/regression_diagnosis.md` 中的退步项。

### count_exact（来源 `v1.3_count_pairs`）

- **内容**："two clocks"、"两个钟"这类计数问题，放在恰好有那么多个实例的图上，作为正例。
- **生成方式**：纯规则。
- **适用条件**：2 ≤ n ≤ 5；每个实例的面积至少占图像的 0.5%；不含 crowd 框；排除容易堆在一起的类别。
- **作用**：与 Glance-Data 的 `neg_count`（数量对不上的计数负例）构成最小对，让"有数字"不再能单靠文本判定为"没有"。arch-exp 的实验 5 发现，一个只看文本的探针能把 `neg_count` 判对 98.6%。

### refcoco_zh（来源 `v1.3_refcoco_zh`）

- **内容**：RefCOCO / + / g 训练集表达式的中文翻译，作为长描述正例，框沿用原表达式的框。
- **英文原句不在本数据集中**：因为 RefCOCO 许可的原因，英文原句只在 `manifest/refcoco_zh_pairs.jsonl` 里以引用形式出现（`image_id`、`box`、`dataset`、`en_sha1`），不重新分发原文。训练时，英文原句和中文译句组成一致性组。
- **翻译来源**：
  - 约 1.2 万条由 Claude Opus 5.5 翻译（订阅 CLI）。
  - 其余由 Qwen3.8-27B-FP8 翻译。这部分要通过往返过滤才保留：先回译成英文，再要求 MiniLM 余弦相似度 ≥ 0.75、原句里的颜色 / 数量 / 方位词全部保留、主名词保留。
- **质量**：过滤后的残余错误率，按 Opus 对 500 条的质检：Opus 质检 500 条，判错 14 条，残余错误率 2.8%（`v1.3/raw/qc_opus.jsonl`）。往返过滤共检查 68,238 条，保留 54,731 条；被丢弃的原因是低相似度 10,160 条、属性丢失 3,037 条、主名词丢失 310 条。

### 人写表达回放清单（`manifest/refcoco_replay.jsonl`）

v1.3 训练时回放的 RefCOCO / + / g 人写表达，以及 v1 的困难负例。只列 id，不含原文：`image_id`、`present`、`box`、`dataset`、`neg`、`text_sha1`。

## 已知问题

- 泛称词的日常含义比 COCO 类别宽。例如"creature"可以指人，"accessory"可以指护腕。所以泛称负例里约有 1/8 的条目存在歧义，人工抽检时发现的错误也都属于这一类。
- 盲验证只能看到已标注的物体编号。图中没有标注的物体（比如笔、墙上的灯）偶尔会让负例出错。
- v1.3 负例的生成在 4,096 张图处提前停止，原因是 A100 吞吐太低。因此关系负例的数量少于计划；训练时通过下采样关系正例，维持正负 1:1。
- arch-exp 的实验 5 显示，Glance-Data 的模板化负例能被一个只看文本的分类器识别。本数据集里的计数最小对用来抵消这一点。使用 B 组（诊断集）结果时，应同时报告一个只看文本的基线。

## 许可

- 图像：COCO（CC BY 4.0，具体许可以各图片的 Flickr 来源为准）。
- 新写的句子和标签：与 Glance-Data 相同，CC BY-NC 4.0。
- RefCOCO 系列的原文不在本数据集中重新分发，只以 id 或哈希引用。
- 本数据集为组内私有，是否公开由用户决定。

## 复现

构建所用 commit：0fdd840

```bash
# 1. v1.2 无类别词数据：scripts/v12/{catfree_scenes,catfree_gen,catfree_build}.py（见 scripts/v12/vm_chain.sh）
# 2. v1.3 负例
python scripts/v13/scenes_v13.py --ann ann --gd glance_data --exclude_ids probe_ids.json,robust_ids.json --n 30000 --out scenes_v13.jsonl
python scripts/v12/catfree_gen.py --scenes scenes_v13_10k.jsonl --img_root coco/train2014 --out gen_v13.jsonl --n_neg 2
python scripts/v12/catfree_build.py --scenes scenes_v13_10k.jsonl --gen gen_v13.jsonl --out_dir out --n_pos 0 --n_neg 200000
# 3. 计数最小对
python scripts/v13/count_pairs.py --ann ann --exclude_ids eval_image_ids.json --out count_pos.jsonl
# 4. 中文翻译 + 往返过滤
python scripts/v13/translate_zh.py ...; python scripts/v13/translate_opus.py ...; python scripts/v13/backtrans.py ...; python scripts/v13/filter_bt.py ...
# 5. 打包
python scripts/glance_data_plus/build_plus.py --gd_full ... --commit 0fdd840 --out gdplus
```
