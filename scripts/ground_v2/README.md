# ground_v2：v1.2 grounding 数据流水线

> **现状（2026-11-18）。** 组里的完整数据集 `Vision-Decision/Glance-Data` 已经发布：24,406 张图，183 万题，抽检 101/112。它覆盖了下面设计的大部分题型，所以 v1.2 改为直接使用它：
>
> - `convert_glance_data.py` 把它转成 Glance 的记录格式。
> - 规则补上它缺的一类：单实例上的方位词和最高级正例 `pos_pos_sole`，数量与它的 `neg_position` 按 1:1 配平。
> - 转换结果推到私有 HF `Vision-Decision/glance-ground-v2`，有 `full`（多框 head）、`single`（`n_targets` ≤ 1）、`diagnostic` 和 `diagnostic_pos_sole` 四个配置。
> - `eval_diag.py` 和 `vm_eval_diag.sh` 在它的 300 图诊断集上按题型 × 语言评测 v1 和 v1.1。
>
> 下文的自建三模块流水线（`build_scenes.py`、`gen_llm.py`、`build_questions.py`）保留在仓库里，留作补数据用：它在 200 图试点上验证过，结果见文末。

v1 的 box 数据只有 RefCOCO 风格的短语。它带来四个已经实测到的毛病：

- 中等长度、口语、中文、功能式说法掉分（"ketchup" 0.976，"the ketchup bottle" 0.362；"保存成 PNG 的按钮" 0.11）。
- 裸类别名在多实例图上只框最大的一个（86.5%），gRefCOCO 多目标并集只中 41.3%。
- 单实例上的方位词和最高级被拒（只有一只猫时问 "the leftmost cat"，86.5% 判为不存在）。
- 数量、大小错配的误接受偏高（14.1% / 13.5%）。

这条流水线照搬组里 "Grounding 数据 Pipeline" 的三段式设计：**所有答案由规则从标注里算出，LLM 只负责写句子**。我们在它的基础上加了一步盲验证：生成句子后，另起一次调用让模型在不知道目标的情况下重新指认，结果对不上就丢掉。

## 三个模块

| 模块 | 脚本 | 做什么 |
|---|---|---|
| 1 规则 | `build_scenes.py` | 按图合并 COCO 实例与 caption、RefCOCO/+/g train 表达式、gRefCOCO train（含多目标和无目标）。每张图选最多 6 个物体编号让 LLM 描述（按类别轮转，优先有人工表达式和面积大的），再选最多 3 组多目标：同类的左/右两个，或两个不同类的唯一物体。同时标出 `marginal`（太小，或贴边且 < 1%）。 |
| 2 LLM | `gen_llm.py` | `deepseek-v4.1-flash`（Ollama 云 API，**有视觉**），输入带编号框的图和原图。每个物体写 short / medium / long / colloquial / zh_short / zh / function / zh_function，记录观察到的颜色和动作，再写 colour_neg / action_neg（只改颜色或动作）。每组写一条点名全部目标的英文和中文。随后**第二次调用**拿到打乱、去掉目标的全部句子，回答每句指哪些编号（`[]` 表示什么都不指，`"other"` 表示指未编号物体或有歧义）。 |
| 3 规则 | `build_questions.py` | 按标注直接出题（框、计数、类别有无都是精确的），加上过滤。 |

辅助脚本：`render_samples.py`（抽检图），`check_decontam.py`（去污染断言），`upload.py`（推到私有 HF），`remote_run.py`（在 gen-host 上后台运行；key 经 ssh stdin 进入进程环境，不落盘、不进命令行），`debug_call.py`（单次调用排错）。`lexicon.py` 是类别的中英文名、同义词、复数和相近类别；其中的同义词表抄自 `scripts/robust/cat_names.py`（S3 sprint，写这里时还没提交），两边都合进 main 以后应改成直接 import。

## 题型（`qtype`）

| 类 | qtype | 来源 | 答案 |
|---|---|---|---|
| 正例 | `pos_short` `pos_mid` `pos_long` `pos_colloq` `pos_zh` `pos_func` | LLM，经验证 | 一个框 |
| | `pos_multi` | LLM 组描述，点名每个目标，经验证 | 2–3 个框 |
| | `pos_multi_human` | gRefCOCO train 多目标（v0/v1 丢掉的那部分） | 全部目标框 |
| | `pos_cat` | 裸类别名，图中唯一实例（中英） | 一个框 |
| | `pos_cat_all` | 裸类别名或复数，多实例 | **全部实例** |
| | `pos_all` | "all the X" / "every X" / "both X" / "the three X" / "所有的X" | 全部实例 |
| | `pos_pos_sole` | 唯一实例上的方位词或最高级："the leftmost cat"、"the biggest cat"、"最左边的猫" | 一个框 |
| | `pos_pos_multi` | 多实例中的最左 / 最右 / 最大，要求几何上有明显间隔 | 一个框 |
| | `pos_human` | RefCOCO/+/g train 表达式（每物体 ≤ 2 条，每图 ≤ 12 条） | 一个框 |
| 负例 | `neg_colour` `neg_action` | LLM 只改颜色或动作，验证结果为"什么都不指" | 不存在 |
| | `neg_count` | "the six zebras"（实际 3 只），"three cats"（实际 1 只）；数字至少比实际多 2 | 不存在 |
| | `neg_similar` | 图中没有的相近类别，裸名或替换已验证 medium 句里的名词（"the brown cat on the couch"，实际是狗） | 不存在 |
| | `neg_absent` | 图中没有的远类别 | 不存在（easy） |
| | `neg_position` | 唯一实例明显在一侧（中心 < 0.3 且整体在左半边），问另一侧 | 不存在 |
| | `neg_size` | 占画面 ≥ 30% 的物体问 "the tiny X" | 不存在 |
| | `neg_human` | gRefCOCO train 无目标表达式 | 不存在 |

组里的方案把 LLM 写的"改位置"负例判为不可靠，这里没有用；`neg_position` 完全由几何规则生成，阈值很严。

## 过滤（组里三条加上我们的）

- **类别、计数、方位类题目**只用"实例集合可靠"的类别：没有 crowd，没有 marginal 实例，n ≤ 8，不是易成堆、标不全的类别（book、banana、carrot、broccoli、apple、orange、donut），实例框之间没有大面积嵌套（交集 / 较小框 > 0.6）。前两条对应组里的"物体大部分被裁掉就跳过"，后两条来自本仓库的抽检。
- **多目标句必须点到每个目标**：每个类别名或同义词都要出现（人另有角色词表），同类组还必须有数量词。
- **short 必须在图中唯一**：同类有多个实例时不能只是裸类别名；两个物体的 short 不能相同；最后再由盲验证兜底。
- **function 句**不能出现类别名、同义词或上位词，必须带功能线索（to / for / used / that… / 用 / 能 / 装…），人和动物不出 function 句。
- **action 负例**只给人和动物（静态照片看不出车在不在动），而且不能保留原来的动作词（像 "curled up running" 这种自相矛盾的句子要丢掉）。
- 语言检查（zh 字段必须有汉字，en 字段不能有），不能提到画上去的编号框，同一句话在同一张图里不能对应两个不同答案。

## 记录格式

与 `dv/data/mixed.py` 的 box 题一致，另加多框 head 约定的字段：

```json
{"src": "ground_v2", "img": "coco/train2014/COCO_train2014_000000276017.jpg", "wh": [640, 480], "max_vis": 640,
 "plen": 305, "image_id": 276017,
 "questions": [
   {"type": "box", "text": "all three zebras", "present": true, "box": null,
    "boxes": [[x1,y1,x2,y2], [..], [..]], "n_targets": 3, "qtype": "pos_all", "lang": "en", "neg": null, "src": "ground_v2"},
   {"type": "box", "text": "the leftmost cat", "present": true, "box": [x1,y1,x2,y2], "boxes": [[x1,y1,x2,y2]],
    "n_targets": 1, "qtype": "pos_pos_sole", "lang": "en", "neg": null, "src": "ground_v2"},
   {"type": "box", "text": "the six zebras", "present": false, "box": null, "boxes": [], "n_targets": 0,
    "qtype": "neg_count", "lang": "en", "neg": "hard", "src": "ground_v2"}]}
```

- 坐标是原图像素，格式 xyxy。
- `box` 只在单目标时有值，以兼容 v1 的单框 head。多目标题 `box = null`、`present = true`，旧 head 在这类题上只训练存在性。
- 训练记录每条最多 48 题；测试集每张图一条记录，包含该图全部题目，另附 `url` 和 `license`。

## 去污染

训练池 = COCO train2014，去掉以下图像：

- 所有评测 split 的图：RefCOCO/+/g val/test*，gRefCOCO val/testA/testB（沿用 `dv/data/prep.py` 的规则）；
- **全部 COCO val2017**：val2017 复用 train2014 的图像 ID，歧义 / 属性探针和本测试集都在这里面；
- `--exclude_ids` 指定的图。

`Vision-Decision/glance-robust-ref` 建在 RefCOCO/+/g val 的图上（见 `scripts/robust/build_src.py`），所以已经被排除；但写这里时 HF 上还读不到这个 repo，`check_decontam.py` 会在能读到的时候再直接核对一遍。

测试集取自 val2017，去掉探针用过的 151 张图，所以 v1、v1.1（训练用的自然图只有 RefCOCO-train）和 v1.2 都没见过这些图。每次构建后运行 `check_decontam.py`，有任何重叠就报错退出。

## 运行

```bash
# gen-host: ~/gv2/{ann,img,code,.venv}
python scripts/ground_v2/build_scenes.py --ann ~/gv2/ann --split train --n 0 --out ~/gv2/scenes_train.jsonl
python3 scripts/ground_v2/remote_run.py ~/gv2/gen_train.log -- python scripts/ground_v2/gen_llm.py \
    --scenes ~/gv2/scenes_train.jsonl --img_root ~/gv2/img --out ~/gv2/gen_train.jsonl --workers 8
python scripts/ground_v2/build_questions.py --scenes ~/gv2/scenes_train.jsonl --gen ~/gv2/gen_train.jsonl --require_gen --out_dir ~/gv2/out_train
python scripts/ground_v2/check_decontam.py --ann ~/gv2/ann --train ~/gv2/scenes_train.jsonl --test ~/gv2/scenes_test.jsonl --probe_ids ~/gv2/probe_ids.json
python scripts/ground_v2/render_samples.py --data ~/gv2/out_train/train.jsonl --img_root ~/gv2/img --out_dir ~/gv2/sheets --k 8
```

## 结果

### A. glance-ground-v2（Glance-Data 转换版，v1.2 实际使用）

| 配置 | 记录 | 题目 | 说明 |
|---|---:|---:|---|
| `full` | 51,177 | 1,894,336 | Glance-Data 183.1 万题，加 63,190 条 `pos_pos_sole` |
| `single` | 45,027 | 1,620,520 | 只保留 `n_targets` ≤ 1 的题 |
| `diagnostic` | 300 | 23,578 | Glance-Data 诊断集（refcoco_val 的图），只用于评测 |
| `diagnostic_pos_sole` | 160 | 764 | 同一批图上的单实例方位 / 最高级正例 |

- `pos_pos_sole` 与 `neg_position` 的比例为 63,190 : 63,198。
- `pos_pos_sole` 抽检 8/8 正确：比如 "the knife on the left"（图里只有一把刀）、"最右边的键盘"、"唯一的那个披萨"、"the rightmost horse"。
- 去污染：训练图与 RefCOCO/+/g/gRefCOCO 评测图、COCO val2017、探针图、glance-robust-ref、诊断集的交集都是 0。

### C. 基线：v1 / v1.1 在诊断集上的表现（2026-11-18，Colab G4，原始 P(present)，阈值 0.5）

- **评测集。** Glance-Data 诊断集（300 张 refcoco_val 图，23,578 题），加上同一批图上的 764 条 `pos_pos_sole`。
- **指标。**
  - 单目标题：门控 Acc@0.5，即 P ≥ 0.5 且与第一个目标的 IoU ≥ 0.5；括号里是 recall（P ≥ 0.5）。
  - 负例：N-acc（P < 0.5）。
  - 多目标题：recall / 命中任一目标（门控）/ 与并集框 IoU ≥ 0.5（门控）。
- **原始数据。** 每题的输出和汇总在 `Vision-Decision/glance-ground-v2` 的 `eval/diag_2026-11-18/`，汇总也在本仓库的 `outputs/ground_v2/diag/`。表格由 `diag_table.py` 生成。

| kind | 指标 | n（en/zh） | v1 en | v1 zh | v1.1 en | v1.1 zh |
|---|---|---|---|---|---|---|
| short | Acc@0.5（recall） | 1378/1379 | 78.2 (96.4) | 77.1 (96.5) | 78.2 (95.8) | 77.5 (96.3) |
| medium | Acc@0.5（recall） | 1370/1379 | 81.4 (94.5) | 81.3 (96.0) | 82.8 (94.3) | 81.9 (94.9) |
| long | Acc@0.5（recall） | 1368/1364 | 81.8 (94.6) | 81.9 (95.7) | 83.4 (94.5) | 83.5 (95.3) |
| casual | Acc@0.5（recall） | 1378/1382 | 79.6 (94.6) | 79.7 (94.9) | 80.0 (93.4) | 79.0 (93.4) |
| cat_only | Acc@0.5（recall） | 308/218 | 55.2 (75.0) | 42.7 (78.4) | 49.0 (69.8) | 37.2 (77.5) |
| **pos_pos_sole** | Acc@0.5（recall） | 382/382 | **24.6 (31.7)** | **28.0 (36.1)** | **24.1 (32.2)** | **29.1 (35.3)** |
| function（单目标） | Acc@0.5（recall） | 178/178 | 44.9 (85.4) | 46.1 (73.0) | 44.9 (71.3) | 45.5 (65.7) |
| group | recall / hit / ∪ | 530/535 | 98.5 / 67.2 / 65.8 | 99.3 / 70.5 / 67.1 | 98.9 / 63.4 / 80.8 | 98.7 / 67.7 / 78.3 |
| cat_all | recall / hit / ∪ | 462/462 | 84.0 / 36.8 / 43.7 | 80.7 / 39.2 / 42.9 | 88.5 / 33.5 / 60.6 | 85.5 / 39.6 / 58.9 |
| function（多目标） | recall / hit / ∪ | 367/366 | 87.2 / 49.0 / 19.3 | 83.1 / 49.7 / 19.4 | 76.6 / 48.5 / 25.3 | 77.0 / 53.6 / 23.8 |
| gref_multi | recall / hit / ∪ | 1014/– | 94.6 / 70.5 / 36.7 | – | 96.1 / 75.5 / 41.0 | – |
| hneg_attribute | N-acc | 1331/1330 | 68.4 | 66.2 | 70.5 | 66.2 |
| hneg_relation | N-acc | 897/934 | 40.7 | 30.1 | 33.8 | 26.3 |
| neg_count | N-acc | 864/864 | 69.0 | 64.4 | **49.1** | **44.8** |
| neg_position | N-acc | 356/354 | 90.7 | 87.0 | 90.7 | 84.5 |
| neg_near | N-acc | 516/516 | 95.3 | 89.9 | 91.5 | 89.0 |

**读法。**

1. **单实例方位 / 最高级正例是最大的洞**：只有 32–36% 被判为存在，`neg_position` 的 N-acc 却有 85–91%。两者合起来看，模型学到的是"方位词意味着不存在"，和探针结论一致。v1.2 加的 63k 条 `pos_pos_sole` 正是针对这一点。
2. **裸类别名（cat_only）只有 37–55%**：recall 70–78%，单实例的裸类别名有四分之一被拒。组里报告 Qwen 在这一项是 40%。
3. **单框 head 的多目标上限。** 门控后 cat_all 命中并集只有 43–61%，function 多目标只有 19–25%，需要多框 head。v1.1 的并集命中明显高于 v1（cat_all 43 → 60），说明它更倾向于框住整组。
4. **难负例。**
   - 属性错（hneg_attribute）N-acc 66–70%，比探针里颜色单独换的 99% 难得多：Glance-Data 的改动藏在中长句里。
   - 动作 / 关系错（hneg_relation）只有 26–41%。
   - 数量错在 v1.1 上从 64–69% **退到 45–49%**。
   - 组里的 Qwen 在同类上为：颜色错 16%，动作错 8%，数量错 57%。
5. **中英文差距小**（短 / 中 / 长 / casual 都在 1 个点以内），只有 cat_only（en 55 vs zh 43）和 hneg_relation（en 41 vs zh 30）中文明显弱。

### B. 自建流水线试点（195 张 train2014 图，加上测试集 129/300 张）

- **规模。** 每张图约 43 题，其中约 30% 是中文、30% 是负例。
- **LLM 句子的保留率**（规则过滤和盲验证之后）：
  - colour_neg 96%，action_neg 87–90%；
  - mid / long / colloquial / zh 81–83%，short 75–78%；
  - function 56–65%，multi 40–56%。
- **DeepSeek 用量。** 每张图约 2.75k 输入 token、0.99k 输出 token，分两次调用。Ollama 云是订阅制，不按 token 计费。按 DeepSeek 官方 API 价目估算（未核实 v4.1-flash 的单价），每张图约 ¥0.008。
- **吞吐。** 单个 key 有并发上限，实测每分钟 15–19 张图。2026-11-18 13:00 前后 key 触发了 **session usage limit**，所有请求都返回 429，测试集只生成了 129/300 张，随后转向 Glance-Data。

**人工抽检。** 每类 8 条，在渲染出的图和框上逐条判断：

| qtype | 第 1 轮 | 修复后 | 主要错误与修复 |
|---|---|---|---|
| pos_short | 7/8 | 6/8（测试集） | 拥挤同类（一院子花瓶）和贴边碎片仍会出现 → 新规则：只描述同类 ≤ 6、无 crowd、不是"贴边且 < 3%"的物体 |
| pos_mid | 8/8 | 6/8（测试集） | 同上，两例都被新规则覆盖 |
| pos_long | 8/8 | 6/8（测试集） | 同上，另有一例图太暗 |
| pos_colloq | 8/8 | | |
| pos_zh | 8/8 | | |
| pos_func | 7/8 | 7/8 | 多个杯子时问"用来喝水的东西"有歧义；新增功能线索检查，人和动物不出 function 句 |
| pos_multi | 8/8 | | |
| pos_cat | 8/8 | | |
| pos_cat_all | 7/8 | 7/8 | 书堆、胡萝卜堆标注不全 → 排除易成堆的类别；剩下的错误是 COCO 把锅标成了 bowl |
| pos_all | 6/8 | 7/8 | 同上；另外排除了嵌套框 |
| pos_pos_sole | 7/8 | | COCO 把沙发标成了 bed |
| pos_pos_multi | 8/8 | | |
| pos_human | 8/8 | | |
| pos_multi_human | 8/8 | | |
| neg_colour | 8/8 | | |
| neg_action | 6/8 | 8/8（新 prompt） | 自相矛盾的叠加动作（"curled up running"），以及车辆的运动状态无法从静态图判断 → 只给人和动物出，要求替换动作，并用规则检查 |
| neg_count | **5/8** | 8/8 | 堆状物体、背景里没标注的实例 → 数字至少比实际多 2，并排除易成堆的类别 |
| neg_similar | 7/8 | | 柜台上有一把没标注的剪刀 |
| neg_absent | 8/8 | | |
| neg_position | 7/8 | | |
| neg_size | 8/8 | | |
| neg_human | 8/8 | | |
| **合计** | 163/176（92.6%） | 164/176（93.2%，保守口径） | 没有低于 6/8 的类型 |

"修复后"一列是修复后重新抽样的结果。短 / 中 / 长三类的测试集抽检早于"可描述性"规则，所以那三格是保守数字。
