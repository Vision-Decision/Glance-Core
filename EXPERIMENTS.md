# Glance 实验索引

Glance 是基于 Qwen3.5-4B 的类型化决策模型：一张图只做一次 prefill，之后可以连续回答多个问题。框头会给出校准后的存在概率 `P(present)`，IoU 头给出框质量 κ̂。

这份索引列出项目里做过的全部实验。每条记录写明：代码入口和命令、数据来源、检查点位置、结果文件、关键数字、状态和对应的提交。目标是只靠 GitHub 组织 `Vision-Decision` 下的仓库和 Hugging Face 组织（`Vision-Decision`）就能复现每个实验。少数复现不了的，单独列在文末的「缺口」一节，并写明原因。

几点约定：

- **仓库**：本索引写于单仓库时代，路径按原目录写。现在各目录分属下面这些仓库，查找时按路径前缀对照。少数研究记录不在本索引中。

  | 路径前缀 | 仓库 |
  |---|---|
  | `dv/`、`train.py`、`eval.py`、`configs/`、`scripts/`、`ablation/`、`gen_v1/`、`arch/`、`evals/`、`di/`、其余 `outputs/` | [glance-core](https://github.com/Vision-Decision/glance-core)（本仓库） |
    | `speed/`、`scan/`、`outputs/scan/`、`outputs/sprint_speed/`、`outputs/engine_opt/` | [glance-engine](https://github.com/Vision-Decision/glance-engine) |
    | `video/`、`scripts/video/` | [glance-video](https://github.com/Vision-Decision/glance-video) |
    
- **关键数字**都取自结果文件，括号里注明文件和字段。
- **VenusBench-GD** 一律使用官方 UI-Venus 评测器的分数：v1 Overall **40.06**，v1.1 **40.68**（`outputs/sprint_abs/v1/tables.md`、`outputs/sprint_abs/v1.1/tables.md` 的 `raw_gated` 行）。早期自写脚本算出的 47.3 和 52.3 已经作废，不要再引用。
- **HF 位置**写成 `仓库:路径`。`Vision-Decision/glance-train-ckpts-legacy` 等仓库原来在个人账号下，现已搬到组织；旧 id 读取时会重定向，但脚本和文档一律写组织下的新 id。
- **分支**：不在 main 上的代码，会注明所在分支。下面这些分支都已推到 origin。

---

## 0. 环境与数据

### 0.1 运行环境

- **GPU**：训练和评测都跑在 Colab G4（RTX PRO 6000 Blackwell，96 GB）上，通过 `colab-pool` 开机。在全新 VM 上执行 `bash /content/dv/scripts/colab_setup.sh`，它会安装 `transformers==5.23.0`、`flash-linear-attention==0.6.0` 和 `peft`，然后下载 `Qwen/Qwen3.5-4B` 和 COCO train2014。
- **同步代码**：用 `scripts/sync.sh <session>` 把代码和 `data/prep` 推到 VM。
- **CPU 任务**：数据合成、打分、出表在 gen-host 上跑（`ssh gen-host`，30 核，无 GPU）。
- **Video-Index 沙箱**：在 gen-host 的 `~/videoindex` 下运行，详见 `video/sandbox/README.md`。

### 0.2 Hugging Face 认证

组织下的数据集和检查点大多是私有的，需要先登录一个有 `Vision-Decision` 组织读权限的账号：

```bash
hf auth login           # 或者 export HF_TOKEN=...（不要写进仓库）
hf download Vision-Decision/glance-train-ckpts v1.2/ckpt_final.pt --local-dir /content/ckpt
hf download Vision-Decision/glance-train-data --repo-type dataset --local-dir /content/gtd
```

### 0.3 HF 仓库一览

| 仓库 | 类型 | 内容 |
|---|---|---|
| `Vision-Decision/Glance-4B-preview` | model | 公开发布的权重（adapter + heads），对应 **v1**（`glance_config.json` 中 `version: v1`，权重提交为 `b6237b4`） |
| `Vision-Decision/glance-train-ckpts` | model | v1.2、多框 pilot、scan 短读出、Glance-Video、生成式基线、ar_iou 特征、leak4b、gv0 结果 |
| `Vision-Decision/glance-train-ckpts-legacy`（原个人仓库，已搬到组织） | model | v0、27B、ablation 全部、v1、v1.1（含 release 导出和逐行输出）、sprint_abs 逐行输出、sprint_robust、sprint_gui ft1、robust_eval、多框 pilot 的 base/heads/jointnb 评测 |
| `Vision-Decision/glance-train-data` | dataset | 本次新建。存放原先只在本机、gen-host 或会话临时目录里的中间数据：v1 自然图像题池、多框 pilot 数据、S3 增强题池、v1.2 catfree 场景文件、scan 蒸馏清单 |
| `Vision-Decision/glance-ground-v2` | dataset | 由 Glance-Data 转换来的训练数据（`full/`、`single/`）、诊断集、v1.2 catfree 数据 |
| `Vision-Decision/Glance-Data` | dataset | 同组同学的 grounding 数据（parquet） |
| `Vision-Decision/glance-robust-ref` | dataset | 描述偏移评测集（中文、改写、功能式、模糊） |
| `Vision-Decision/guisyn-v0` | dataset | 合成 GUI 截图、可访问性树和 grounding 题 |
| `Vision-Decision/glance-video-v0` | dataset | Glance-Video 的片段 shard、records、去污染表、回放数据 |
| `Vision-Decision/scan-pilot-data` | dataset | Charades-STA 测试帧（1 fps 和 2 fps）和训练片段 tar |
| `Vision-Decision/jev-runs` | dataset | Jev 1.14 的原始响应和汇总 |
| `Vision-Decision/di-runs` | dataset | Decision Index 文本榜和视觉榜的逐行结果 |

### 0.4 公开数据（只记录来源，不转存）

- **RefCOCO / RefCOCO+ / RefCOCOg / gRefCOCO**：标注从 HF `jxu124/{refcoco,refcocoplus,refcocog}` 和 `FudanCVL/gRefCOCO` 下载，图像用 COCO train2014。运行 `python dv/data/prep.py` 生成 `data/prep/`，这份产物已提交进 git。
- **v1 的文本、GUI、指点数据**：用 `scripts/v1/fetch.sh` 下载，v1.1 另加 `scripts/v1/fetch_v11.sh`。数据源包括 OpenDecisions、DecisionBank、OS-Atlas、AGUVIS、GroundCUA、Jedi、PixMo 等。
- **GUI 评测集**（ScreenSpot-v2、ScreenSpot-Pro、VenusBench-GD）：由 `evals/gui_data.py` 下载。
- **Charades、K700、Countix、RepCount-A**：下载和切片脚本是 `scan/charades_prep.py`，以及 `glance-video` 分支上 `scripts/video/` 下的脚本。
- **Video-Index**：许可只允许用于评测，所以视频和帧都**不转存**，也不进任何 HF 仓库。下载用 `video/sandbox/fetch.sh`，它会执行 `hf download Video-Index/Video-Index-v1-1 --repo-type dataset`。

---

## 1. 总表

| # | 区域 | 实验 | 一句话 | 状态 | 提交 |
|---|---|---|---|---|---|
| 1 | 精确性 | equiv-toy | 共享前缀前向与 HF 参考前向逐元素一致（小配置，CPU） | 完成（输出未存档） | 660553c |
| 2 | 精确性 | leak4b | 在 4B 上测共享前缀的精确性，以及块对角打包是否泄漏 | 完成 | 97905f8, e312aa1 |
| 3 | 精确性 | 27b-leak | 早期 27B 泄漏测试 | 已废弃（被 leak4b 取代） | 942e648 |
| 4 | 精确性 | speed-equiv | 各推理变体相对 fp32 的决策一致率 | 完成 | 83f2090, 04a775a |
| 5 | 框头 | v0 | 4B 框原语：存在门控 + patch 中心 + 4 条边 + IoU 头 | 完成 | 660553c |
| 6 | 框头 | 27b | 用 v0 配方训 27B | 完成 | 942e648 |
| 7 | 框头 | v1 | 通用类型化决策模型（choice/score/noul + box） | 完成 | 942e648 |
| 8 | 框头 | ablation | 受控实验：同一底座只换输出头，共 7 个 run | 完成 | 942e648, 1b2e484 |
| 9 | 框头 | ablation-v2 | 精确 MAP、修好的 beam、双旋钮、RCPS、公平延迟、换种子 | 完成（种子只做了一部分） | 09c28ed, 8b8d756 |
| 10 | 框头 | ablation-v3 / ar_iou | AR-digits 加训练出的 IoU 头，两种校准协议，并做框精度对齐 | 完成 | 54dff43, e1bb5e6, 08d4d89 |
| 11 | 框头 | sprint-metrics | 置信区间、配对检验、替代协议 | 完成 | 08503e3, c13c782 |
| 12 | 框头 | gen-v1 | 生成式基线：同样的数据和底座，只把输出换成文本 | **进行中** | 1c36e0e, 9d23775, 148cb56 |
| 17 | S4 | sprint-speed | Glance 引擎与 vLLM/SGLang 公平对比延迟 | 完成 | 04a775a, 3f7c03a |
| 18 | S3 | robust-ref | 描述偏移评测集 | 完成 | cbe7792 |
| 19 | S3 | robust-diag | 诊断：问题出在训练数据还是读出方式 | 完成 | ca506bc |
| 20 | S3 | robust-recipes | 6 种续训配方（c0/r1/r2/r3/d_heads/a_nofunc） | 完成 | c7ab75a, d1291ec |
| 21 | 官方协议 | abs-v1 / abs-v1.1 | 官方 GREC 和 VenusBench-GD 评测器、校准、风险控制 | 完成 | d5beeb1, 7e022cc |
| 22 | 官方协议 | abs-hnft | 难负例续训和多目标续训 | 已废弃（负结果） | 7e022cc |
| 23 | 官方协议 | grec-curve | v1 的 GREC Pr–N-acc 阈值曲线 | 完成 | af9fad6 |
| 24 | 官方协议 | s1-gui | ScreenSpot-Pro 冲分：原生分辨率、zoom、ft1/ft2 | 完成 | 483dcb8, f87fc74 |
| 25 | 官方协议 | sspro-base | 底座在同分辨率下跑 ScreenSpot-Pro | 完成 | 5413b4d, bf6bc70 |
| 26 | 官方协议 | xdomain-v0 | v0 和底座的跨域 GUI 逐行输出 | 完成 | f804073, 6ef37dd |
| 27 | Jev | jev | Jev 1.14 与 Glance v1 在同一批题上配对比较 | 完成 | b93ea73 |
| 28 | Jev | di | Decision Index 文本榜和视觉榜 | 完成 | 08ce8f7, 28b8a02 |
| 29 | 数据 | gv2-convert | Glance-Data 转成 glance-ground-v2 | 完成 | bc1abd7, edc2935 |
| 30 | 数据 | gv2-diag | v1 和 v1.1 在诊断集上的基线 | 完成 | a97a790 |
| 31 | 数据 | gv2-pipeline | 自建三模块数据流水线试点 | 已废弃 | b63de64 |
| 32 | v1.1 | v1.1 | 高分辨率 GUI、更多自然图、修好温度网格 | 完成 | c007689, 27979fb |
| 33 | v1.2 | mb-pilot | 多框读出 pilot | 完成 | de771e3（分支） |
| 34 | v1.2 | v12-catfree | 不含类别词的 grounding 数据 | 完成 | 6c59e74（分支） |
| 35 | v1.2 | v12-train | v1.2 训练和预注册验收 | **进行中**（VM `v12`） | 7bc7624（分支） |
| 36 | v1.2 | v12b | 在 v1.2 上加多框 | **进行中** | 0178398（分支） |
| 37 | scan | scan-engine | 一个问题扫多张图的引擎，以及精确性验证 | 完成 | 5932aef |
| 38 | scan | scan-phase1 | 延迟：缓存、新图、query-first，对比 vLLM | 完成 | 0fb9056 |
| 41 | scan | scan-short-v1 | 把读出块蒸馏成短块 | 完成 | 6d3b9a0 |
| 42 | scan | scan-gv | Glance-Video 片段 pilot 和 short_gv0 | 完成 | e1f8bc1, c2bef8c（分支） |
| 43 | Glance-Video | gv-data | 片段池、去污染、训练 records | 完成 | 87dbca8（分支） |
| 44 | Glance-Video | gv-v0 | 全量训练和离线评测 | 完成 | e83fbd5, e46baec（分支） |
| 45 | Glance-Video | gv-tool | ClipBackend 和 vdmscan 工具 | 完成 | b4ba6d4, 72f9a6b（分支） |
| 47 | Video-Index | vi-sandbox | 本地沙箱，多个规划器加不加 vdm 的对比 | 完成（vdm v0 是负收益） | d4a1b44 |
| 48 | Video-Index | vi-choose | 只用 choose 的下界 | 完成 | d4a1b44 |
| 49 | Video-Index | vi-gate | vdmscan 离线门槛 | 完成（未全部通过） | 5c22442（分支） |
| 50 | Video-Index | vi-scan-paired | DeepSeek 不带工具与加 vdmscan 的配对对比（hard 集） | **进行中** | 2516782（分支） |
| 53 | 博客 | blog / slides | 博客数据、插图和 Video-Index 幻灯片 | 完成 | d45cc7b, abb97c6 |

---

## 2. prefill / GDN 精确性

这一组回答一个问题：共享前缀的前向是否和每个问题单独跑一遍的结果相同。

### 2.1 equiv-toy
- **目的**：在随机初始化的小配置上，验证共享前缀前向等于 HF 对 `prefix+block` 的参考前向，并且各问题块之间互相看不见。
- **代码入口**：
  - `python scripts/test_equivalence.py`（CPU）
  - `python scripts/test_numerics.py Qwen/Qwen3.5-4B`（真实权重，需要 GPU）
- **数据**：脚本内部随机生成。
- **检查点**：不需要。
- **结果文件**：没有存档。README 里写的 fp32 ≈1e-7 只是文字记录，精确性以 leak4b 为准。
- **状态**：完成，但输出未存档。
- **提交**：660553c

### 2.2 leak4b
- **目的**：在 Glance v1.1（4B）上测两件事。一是共享前缀引擎和 HF 参考前向差多少，分 fp32/bf16、fla/纯 torch GDN 几种情况。二是把多个问题块打进一条序列、用块对角 mask 隔开时，块之间会不会泄漏。
- **代码入口**：
  - VM 准备：`scripts/leak4b/setup_vm.sh`
  - 运行：`python scripts/leak4b/run.py --release /content/rel/v1.1/release --data refcoco_val_20x8.jsonl --img_root /content/imgs --out results_raw.json`；纯 torch GDN 版本再加 `--no_fla --dtypes fp32`
  - 汇总：`python scripts/leak4b/summarize.py outputs/leak4b`
- **数据**：`scripts/leak4b/refcoco_val_20x8.jsonl`（20 张图 × 8 条指代表达），图像是 COCO train2014。
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:v1.1/release/heads.safetensors` 和同目录下的 `merged-bf16/`。原始结果的备份在 `Vision-Decision/glance-train-ckpts:baselines/leak4b/results_raw.json`。
- **结果文件**：`outputs/leak4b/summary.md`、`outputs/leak4b/results.json`
- **关键数字**（`outputs/leak4b/results.json`）：
  - 纯 torch GDN、fp32 时，块隐状态相对误差最大 9.93e-07，|ΔP(present)| 最大 1.88e-06。
  - 换成 fla kernel，fp32 误差升到 2.15e-04。HF 自己只换 attention 路径也会差 5.20e-05，说明这部分误差来自 fla 的 Triton kernel，不是共享前缀本身。
  - bf16 下，引擎相对 fp32 的偏差是 0.0066，HF bf16 自身的偏差是 0.0069，两者相当。
  - 块对角打包会泄漏：第 1–7 块的相对误差在 0.16–0.18 之间，加不加 mask 结果一样。
- **状态**：完成。
- **提交**：97905f8、e312aa1

### 2.3 27b-leak（已废弃）
- **代码入口**：`scripts/27b/run_bench.sh`、`scripts/27b/leak_test.py`
- **结果文件**：`outputs/27b/bench/`
- **说明**：这组测试的 harness 有问题，打包前后连前缀本身就差 0.38。所以 39.8、44.1、1.8 这几个数都已作废，由 leak4b 取代。
- **状态**：已废弃。
- **提交**：942e648

### 2.4 speed-equiv
- **目的**：检查速度冲刺里的每个推理变体（合并 LoRA、新引擎、CUDA graphs、融合算子、FP8）和 fp32 参考的决策是否一致、框精度是否保持。
- **代码入口**：`python -m speed.diag_equiv --ckpt /content/ckpt/v1/ckpt_final.pt --out OUT.json --n_img 60`；VM 上由 `speed/vm_main.sh` 调用。
- **数据**：`data/prep/eval_refcoco_val.jsonl` 的前 60 张图。
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:v1/ckpt_final.pt`
- **结果文件**：`outputs/sprint_speed/raw/equiv2_g4.json`。注意 `speed_summary.json` 里的 `summary.g4.equiv` 是空的，数字要看这个文件。
- **关键数字**（`equiv2_g4.json` 的 `vs_fp32`，共 930 个回答）：
  - bf16 发布路径（`ref_lora`）和融合引擎（`eng_fused`）：一致率都是 99.78%，Acc@0.5 都是 94.0。
  - FP8：一致率 99.14%，Acc@0.5 93.6。
- **状态**：完成。
- **提交**：83f2090、04a775a

scan 引擎的精确性放在第 12 节。

---

## 3. 框头与校准

### 3.1 v0
- **目的**：做出 4B 框原语：存在门控 + patch 中心 + 4 条边的分布 + IoU 头，一次 prefill 回答所有框问题。
- **代码入口**（见 README 的 Run 一节）：
  - 训练：`python train.py --cfg configs/v0_4b.json`
  - 评测：`python eval.py --ckpt /content/out/v0/ckpt_final.pt --latency --out /content/out/v0/eval_results.json`
  - 底座基线：`python baseline.py --model Qwen/Qwen3.5-4B --n 500 --out /content/out/v0/baseline_results.json`
  - 出表：`python scripts/make_table.py outputs/v0/eval_results.json outputs/v0/baseline_results.json`
- **数据**：`data/prep/train.jsonl`（已在 git 中），由 `python dv/data/prep.py` 生成。
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:v0/ckpt_final_bf16.pt`
- **结果文件**：`outputs/v0/eval_results.json`、`outputs/v0/baseline_results.json`、`outputs/v0/run_summary.json`
- **关键数字**（`outputs/v0/eval_results.json`）：
  - RefCOCO val Acc@0.5 为 95.0。
  - gRefCOCO-merged：N-acc 77.0，AUROC 96.4，ECE 7.5。
  - 1 题 / 8 题的延迟是 83.7 / 84.7 ms。
- **状态**：完成。
- **提交**：660553c、9c8e407

### 3.2 27b
- **目的**：代码不变，只改配置，把 v0 配方扩到 Qwen3.8-27B。
- **代码入口**：
  - 训练：`bash scripts/27b/train_27b.sh full`；新 VM 上跑完整链路用 `scripts/27b/bootstrap_full.sh`
  - 评测：仓库里没有写明命令，`scripts/27b/wait_eval.sh` 只是轮询 `eval_results.json` 是否生成（见缺口）
- **数据**：同 v0。
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:27b/ckpt_final.pt`
- **结果文件**：`outputs/27b/eval_results.json`
- **关键数字**（`outputs/27b/eval_results.json`）：
  - RefCOCO val 95.8。
  - gRefCOCO-merged：N-acc 77.5，AUROC 96.8。
  - 延迟 230 / 233 ms。
- **状态**：完成。
- **提交**：942e648

### 3.3 v1
- **目的**：做通用的类型化决策模型：在框原语之外加上 choice / score / noul 三种读出，配比扩大到文本、GUI、自然图和指点数据，温度事后拟合。
- **代码入口**（见 README 的 v1 一节）：
  1. 下载：`bash scripts/v1/fetch.sh`
  2. 构建题池：`python -m dv.data.prep_v1 natural`，然后 `bash scripts/v1/prep_all.sh`
  3. 去重：`python -m dv.data.prep_v1 postdedup --out /content/data/prep_v1`
  4. 混合：`python -m dv.data.prep_v1 mix --out /content/data/prep_v1 --total 520000`
  5. 训练：`python train.py --cfg configs/v1_4b.json`
  6. 训练后评测：`scripts/v1/post_train.sh`
- **数据**：
  - 自然图像题池在 `Vision-Decision/glance-train-data:v1_natural/prep_v1/pool_natural.jsonl`。
  - 完整的训练混合是 gen_v1 在同一提交 942e648 上重建的版本：`Vision-Decision/glance-train-ckpts:baselines/gen_v1/data/genv1_prep.tgz`，对应图像在同目录的 `genv1_images.tar`。当年 VM 上的原始混合已经没有了。
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:v1/ckpt_final.pt`；发布版是 `Vision-Decision/Glance-4B-preview:adapter_model.safetensors`。
- **结果文件**：`outputs/v1/eval_results.json`、`outputs/v1/run_summary.json`、`outputs/v1/data_stats/`
- **关键数字**：
  - 8 个 split 的平均 Acc@0.5 为 90.8（`outputs/v1/eval_results.json` 中 8 个 split 的 `acc50` 平均）。
  - gRefCOCO-merged：N-acc 87.4，AUROC 97.9（`outputs/v1/eval_results.json`）。
  - 框题延迟：1 题 85.5 ms，8 题 85.6 ms。
  - 同一文件里的 VenusBench 47.3 已作废，官方分数见第 7 节。
- **状态**：完成。
- **提交**：942e648

### 3.4 ablation：受控输出头实验
- **目的**：底座、数据、步数都相同，只换输出头。对比门控 + 序数头，和四种自回归坐标：digits、hier、atomic-1000、atomic-100。负样本比例分 25% 和 5% 两档。
- **代码入口**：
  - 训练和评测：`bash ablation/remote_queue.sh <run...>`
  - 约束解码和延迟：`ablation/post_queue.sh`
  - 出表：`python ablation/analyze.py outputs/ablation`
  - 5% 负样本数据：`python ablation/make_neg_ratio.py 0.05 data/prep/train.jsonl data/prep/train_neg05.jsonl`
- **数据**：`data/prep/train.jsonl`、`data/prep/train_neg05.jsonl`
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:ablation/ours_neg25_bf16.pt`，其余 6 个 run 在同一目录下，文件名格式为 `<run>_bf16.pt`。
- **结果文件**：`outputs/ablation/main_table.md`、`outputs/ablation/main_table.json`、`outputs/ablation/halluc_vs_negratio.json`
- **关键数字**（`outputs/ablation/main_table.md`）：
  - RefCOCO val Acc@0.5（门控 / digits / hier / atomic-1000 / atomic-100）：95.1 / 94.4 / 94.4 / 93.8 / 94.3。
  - 8 题延迟：门控 84.5 ms，AR 在 232–630 ms 之间。
- **状态**：完成。早期用贪心候选算的 MAP 拒答率已被 v2 取代。
- **提交**：942e648、1b2e484

### 3.5 ablation-v2
- **目的**：补了五件事：
  1. 用精确 MAP 和修好的 beam 搜索，验证拒答率是由格式和解码器决定的。
  2. 双旋钮代价。
  3. 和 vLLM 的公平延迟对比。
  4. RCPS 风险控制。
  5. 换随机种子重跑。
- **代码入口**：
  - 训练和评测：`bash ablation/v2_queue.sh`
  - beam：`bash ablation/v2_beam.sh`
  - 精确 MAP：`python -m ablation.exactmap_ab --ckpt CKPT --out OUT.json --rows ROWS.json`
  - 延迟：`bash ablation/v2_latency.sh`
  - 分析：`python ablation/beamfix_analyze.py`、`python ablation/twoknob_v2.py`、`python ablation/agent_rcps_v2.py`、`python ablation/seeds_v2.py`
- **检查点**：沿用 3.4 的检查点，另加种子 1：`Vision-Decision/glance-train-ckpts-legacy:ablation/ours_neg25_s1_bf16.pt` 和 `Vision-Decision/glance-train-ckpts-legacy:ablation/ar_digits_neg25_s1_bf16.pt`。
- **结果文件**：`outputs/ablation_v2/summary.md`、`outputs/ablation_v2/beam_fix/beamfix_summary.json`、`outputs/ablation_v2/twoknob/twoknob.json`、`outputs/ablation_v2/latency/latency_summary.json`、`outputs/ablation_v2/seeds/seeds.json`
- **关键数字**：
  - 精确 MAP 下对存在目标的拒答率：digits 34.0%，hier 23.1%，atomic-1000 12.3%，atomic-100 4.6%（`beamfix_summary.json`）。
  - 双旋钮代价（c_W=5）：Glance 14.06，AR-digits 18.74，差值 Δ = −4.69，95% CI [−5.87, −3.60]（`twoknob.json`）。
  - Q=1/8/32 时的延迟：Glance 67.0 / 66.7 / 70.2 ms；vLLM 开前缀缓存 173.5 / 218.0 / 331.3 ms（`latency_summary.json`）。
- **状态**：完成。种子实验只做了一部分：`_s2` 和 `neg05_s1` 没有跑完，也没有检查点。
- **提交**：09c28ed、8b8d756

### 3.6 ablation-v3 / ar_iou
- **目的**：补一个基线：冻结 AR-digits，给它训练一个 IoU 头。然后在 P1（交叉拟合）和 P2（留出校准集 + 先验平移）两种协议下比较双旋钮代价，最后在框精度对齐的子集上重新比较。
- **代码入口**：
  - VM 上：`bash ablation/v3_ar_iou.sh`
  - 本地：`python ablation/twoknob_v3.py --out outputs/ablation_v3/ar_iou`，再跑 `python -m ablation.twoknob_v3_boxeq`
- **数据**：由 `ablation/ar_iou_prep.py` 从 `data/prep` 切出。
- **检查点**：模型用 3.4 的 ablation 检查点。中间特征在 `Vision-Decision/glance-train-ckpts:baselines/ar_iou/ar_iou_rows.tgz` 和 `Vision-Decision/glance-train-ckpts:baselines/ar_iou/ar_iou_feats/feats_grefcoco_val_merged.npz`。
- **结果文件**：`outputs/ablation_v3/ar_iou/summary.md`、`outputs/ablation_v3/ar_iou/results.json`、`outputs/ablation_v3/ar_iou/boxeq.json`
- **关键数字**：
  - 存在目标上的 κ AUROC：Glance κ̂ 91.6，AR + IoU 头 88.9，AR 长度归一化置信度 66.0（`results.json`）。
  - 框精度对齐后，c_W=5、P1 协议下差值 Δ = −0.18，95% CI [−0.68, +0.36]，不显著（`boxeq.json`），即两者打平。
- **状态**：完成。
- **提交**：54dff43、e1bb5e6、4534809、08d4d89

### 3.7 sprint-metrics
- **目的**：给已有结果补上 bootstrap 置信区间、McNemar 检验、配对 ΔAURC，以及不调阈值的双旋钮代价。
- **代码入口**：`python outputs/sprint_metrics/scripts/main_stats.py`、`python outputs/sprint_metrics/scripts/ablation_stats.py`、`python outputs/sprint_metrics/scripts/two_knob.py`、`python outputs/sprint_metrics/scripts/xdomain_extra.py <rows_dir> <out_json>`
- **结果文件**：`outputs/sprint_metrics/report.md`、`outputs/sprint_metrics/two_knob.json`
- **关键数字**：不调阈值的双旋钮代价，在缺席率 45%、c_W=5 时降低 33%，差值 Δ = −7.52，95% CI [−8.29, −6.77]（`two_knob.json`）。
- **状态**：完成。分支 `worktree-agent-ad2185ba3ae6bc7f3` 上的同名提交和 main 内容一致，不需要合并。
- **提交**：08503e3、c13c782

### 3.8 gen-v1：生成式基线（进行中）
- **目的**：构造一个只换输出形式的对照。底座、LoRA、训练日程都和 Glance v1 相同，数据也是在 942e648 上重建的同一份 v1 数据，只是把输出改成文本答案。读出方式有三种：p(none) dial、语法约束的贪心框、事后训练的 IoU 头。
- **代码入口**：
  1. 在 gen-host 上重建数据：`bash gen_v1/prep_gcp.sh`
  2. 在 VM 上训练和导出：`STAGE=train bash gen_v1/vm_run.sh`，再跑 `STAGE=dump bash gen_v1/vm_run.sh`
  3. 分析：`python gen_v1/analyze.py --v1 DIR --gen DIR --out outputs/gen_v1`
  4. VenusBench：`gen_v1/venus_official.sh`
- **数据**：`Vision-Decision/glance-train-ckpts:baselines/gen_v1/data/genv1_prep.tgz`，以及同目录的 `genv1_images.tar`。
- **检查点**：`Vision-Decision/glance-train-ckpts:baselines/gen_v1/ckpt_final.pt`
- **结果落点**：逐行输出会推到 `Vision-Decision/glance-train-ckpts:baselines/gen_v1/rows/`；分析结果写到 `outputs/gen_v1/`，目前还没有这个目录。
- **关键数字**：暂无。
- **状态**：**进行中**，在 VM `gen-v1` 上跑，相关代码改动在 main checkout 的 `gen_v1/` 下。
- **提交**：1c36e0e、9d23775、148cb56

---

## 5. 速度（S4）

### 5.1 sprint-speed
- **目的**：把 Glance 推理优化到最快，然后和配置最强的生成式部署公平比较。对照方是 vLLM 0.34 和 SGLang，开启前缀缓存、CUDA graphs、MTP 和 FP8。
- **代码入口**：
  - VM 准备：`speed/setup_vm.sh`
  - 主测量：`bash speed/vm_main.sh GPU_TAG all`
  - G4 最终一轮：`speed/vm_g4final.sh`
  - L4：`speed/vm_l4.sh`
  - 汇总：`speed/collect.sh`，再跑 `python speed/report.py --raw outputs/sprint_speed/raw --out outputs/sprint_speed`
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:v1/ckpt_final.pt`
- **结果文件**：`outputs/sprint_speed/summary.md`、`outputs/sprint_speed/tables.md`、`outputs/sprint_speed/speed_summary.json`
- **关键数字**（`outputs/sprint_speed/summary.md`）：
  - G4 上 Q=1/8/32：Glance bf16 为 43 / 45 / 61 ms，vLLM FP8+MTP 为 99 / 134 / 229 ms，Glance 快 2.3 / 3.0 / 3.8 倍。
  - 追问（前缀已缓存）：11–14 ms，vLLM 78–113 ms。
  - L4 上 Q=32：快 5.4 倍。
  - 适用范围：如果一次调用以 prefill 为主（4k token 的截图，每张只问一个问题），Glance 持平或更慢。
- **状态**：完成。
- **提交**：632405c、04a775a、3f7c03a、b5f5262

---

## 6. 鲁棒性（S3）

### 6.1 robust-ref：描述偏移评测集
- **目的**：同一个目标配上中文、改写、功能式、模糊四类描述，每类都有「目标不存在」的变体。用来测换一种说法后模型会不会误判为不存在，以及高置信错误的比例。
- **代码入口**：
  - 构建：`scripts/robust/build_src.py` → `scripts/robust/gen_text.py` → `scripts/robust/finalize.py` → `scripts/robust/make_eval.py`
  - 评测：`bash scripts/robust/run_eval.sh MODEL CKPT`
- **数据**：`Vision-Decision/glance-robust-ref:data/zh.jsonl`，其余三类在同一目录下。评测用的副本在 `data/prep/eval_robust_zh.jsonl` 等文件。
- **结果文件**：`outputs/sprint_robust/summary/tables.md`；逐行输出在 `Vision-Decision/glance-train-ckpts-legacy:robust_eval/scores_all.json`。
- **关键数字**：见 6.3 中独立集一行。
- **状态**：完成。
- **提交**：cbe7792

### 6.2 robust-diag
- **目的**：判断描述偏移下的失分来自训练数据还是读出方式。对比三者：v1、底座走生成式、v1 的 LoRA 走生成式。
- **代码入口**：`bash scripts/sprint_robust/diag.sh`
- **数据**：`data/prep/eval_rg.jsonl`
- **结果文件**：`outputs/sprint_robust/diag_scores.json`、`outputs/sprint_robust/summary/tables.md`
- **关键数字**（`tables.md` 第一张表）：

  | | 偏移后准确率 | 缺席 N-acc | 过度自信错误率 |
  |---|---|---|---|
  | 底座（生成式） | 80.2 | 63.5 | 18.2 |
  | v1 | 75.0 | 88.0 | 3.7 |

- **状态**：完成。
- **提交**：ca506bc

### 6.3 robust-recipes
- **目的**：从 v1 续训 800 步，比较六种修复配方：c0 回放对照、r1 加中文和改写、r2 加全部增强、r3 全部增强加一致性损失、d_heads 只训头、a_nofunc 去掉功能式描述。
- **代码入口**：
  - 每个配方有自己的作业脚本 `scripts/sprint_robust/jobs/<name>.sh`，统一调用 `scripts/sprint_robust/run_recipe.sh`。
  - 对比汇总：`python scripts/sprint_robust/compare.py --runs ... --out outputs/sprint_robust/summary`
- **数据**：
  - 增强题池：`Vision-Decision/glance-train-data:sprint_robust/aug_all_v1.jsonl`，DeepSeek 缓存在 `Vision-Decision/glance-train-data:sprint_robust/aug/cache_expr.jsonl`。这两份本次才从会话临时目录上传。
  - 自然图像题池：`Vision-Decision/glance-train-data:v1_natural/prep_v1/pool_natural.jsonl`
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:sprint_robust/r3_cons/ckpt_final_bf16.pt`，其余五个配方在同级目录下。
- **结果文件**：`outputs/sprint_robust/summary/tables.md`、`outputs/sprint_robust/summary_unseen/tables.md`
- **关键数字**（`summary/tables.md`）：
  - 偏移后准确率：r3 为 83.6，v1 为 75.0。
  - 独立评测集上，r3 的 function_only 从 47.5 升到 73.9。
  - 代价：缺席 N-acc 从 87.2 降到 83.7。
- **状态**：完成。f_v11、r4_lm 等作业只有脚本，没有跑出结果。
- **提交**：f054b4c、c7ab75a、d1291ec

---

## 7. 官方协议：GREC、VenusBench-GD、ScreenSpot

### 7.1 abs-v1 / abs-v1.1
- **目的**：按官方协议重测 v1 和 v1.1。GREC 用阈值 0.7 和 GIoU 匹配；VenusBench-GD 用官方 UI-Venus 评测器。同时做存在概率和 κ̂ 的校准，以及 RCPS 风险控制。
- **代码入口**：
  - 导出逐行输出：`bash scripts/sprint_abs/vm_run.sh TAG CKPT_REPO_PATH`
  - VenusBench 官方打分：`TAG=v1 bash scripts/sprint_abs/chain_venus.sh`。它会 clone `inclusionAI/UI-Venus` 仓库的 VenusBench-GD 分支来打分。
  - 分析：`TAG=v1 bash scripts/sprint_abs/vm_analyze.sh`
  - v1.1 的完整链路：`scripts/sprint_abs/chain_v11.sh`
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:v1/ckpt_final.pt`、`Vision-Decision/glance-train-ckpts-legacy:v1.1/ckpt_final.pt`
- **结果文件**：`outputs/sprint_abs/v1/tables.md`、`outputs/sprint_abs/v1.1/tables.md`、`outputs/sprint_abs/literature.md`；逐行输出在 `Vision-Decision/glance-train-ckpts-legacy:sprint_abs/rows_v1/`。
- **关键数字**：
  - **VenusBench-GD 官方 Overall**：v1 40.06（拒答子任务 81.76），v1.1 40.68（83.76）。
  - GREC 官方口径（阈值 0.7），v1 在 val / testA / testB 上：Pr 71.6 / 47.5 / 51.3，N-acc 91.4 / 85.4 / 81.9。
- **状态**：完成。
- **提交**：282d18d、d5beeb1、7e022cc

### 7.2 abs-hnft / hnft_mt（已废弃）
- **目的**：试两种便宜的续训。hnft 加 GUI 难负例，hnft_mt 再加 gRefCOCO 多目标题。
- **代码入口**：`scripts/sprint_abs/chain_a.sh`（数据由 `scripts/sprint_abs/make_hnft.py` 生成）、`scripts/sprint_abs/chain_mt.sh`
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:sprint_abs/hnft/ckpt_final.pt`、`Vision-Decision/glance-train-ckpts-legacy:sprint_abs/hnft_mt/ckpt_final.pt`
- **结果文件**：`outputs/sprint_abs/compare/`、`outputs/sprint_abs/hnft/tables.md`
- **关键数字**：hnft 让 guisyn 上的 N-acc 从 0.750 升到 0.952，但 VenusBench 上的 N-acc 从 0.818 掉到 0.546。hnft_mt 在 GREC 多目标上的 Pr 仍然是 0。
- **状态**：已废弃，属于负结果。
- **提交**：7e022cc

### 7.3 grec-curve
- **代码入口**：`python3 scripts/sprint_abs/grec_curve.py rows_dir out.json`，输入是 `Vision-Decision/glance-train-ckpts-legacy:sprint_abs/rows_v1/` 下的逐行输出。
- **结果文件**：`outputs/sprint_abs/v1/grec_curve.json`
- **关键数字**：val 上阈值 0.7 时 Pr 71.57、N-acc 91.41。
- **状态**：完成。
- **提交**：af9fad6

### 7.4 s1-gui：ScreenSpot-Pro 冲分
- **目的**：在官方协议下提高 ScreenSpot-Pro、ScreenSpot-v2 和 VenusBench 的分数，手段包括原生分辨率推理、zoom 和 GUI 续训（ft1、ft2）。
- **代码入口**：
  - 评测：`evals/sprint/eval_ckpt.sh HF_PATH TAG`
  - ft1 续训：`evals/sprint/train_run.sh`，配置是 `configs/sprint_gui_ft.json`
  - ft2 续训：`evals/sprint/ft2_chain.sh`
- **数据**：
  - 合成 GUI：`Vision-Decision/guisyn-v0:stats.json`，生成脚本在 `scripts/guisyn/`。
  - 续训用的 GUI 混合数据只在当时的 VM 上（见缺口）。
- **检查点**：v1.1，以及 `Vision-Decision/glance-train-ckpts-legacy:sprint_gui/ft1/ckpt_final.pt`。ft2 没有上传。
- **结果文件**：`outputs/sprint_gui/summary.md`、`outputs/sprint_gui/sprint_gui_results.json`
- **关键数字**（`sprint_gui_results.json`）：

  | 设置 | 分数 |
  |---|---|
  | v1.1，ScreenSpot-Pro 原生分辨率 | 50.66 |
  | 加 zoom | 60.47 |
  | ScreenSpot-v2 | 90.09 |
  | ft1 / ft2（ScreenSpot-Pro） | 49.65 / 50.41，续训没有带来增益 |

- **状态**：完成。
- **提交**：483dcb8、4dadaaa、f87fc74

### 7.5 sspro-base
- **代码入口**：`bash evals/sspro_base/run_vm.sh`
- **结果文件**：`outputs/sspro_base/summary.md`、`outputs/sspro_base/results.json`
- **关键数字**：底座在原生分辨率（最多 16384 token）下得 59.9；在 4000 token 上限下，同一批题得 55.98（`results.json`）。
- **状态**：完成。
- **提交**：1b29e98、5413b4d、bf6bc70

### 7.6 xdomain-v0
- **代码入口**：`bash evals/run_all.sh all`
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:v0/ckpt_final_bf16.pt`
- **结果文件**：`outputs/eval_v0_xdomain/rows/`。剩下 5 个文件之前只在本机，本次已提交。
- **关键数字**：底座在 ScreenSpot-Pro 全部 1,581 题上的平均命中率为 56.0（`base_sspro_4000.jsonl`）。
- **状态**：完成。
- **提交**：dc66d06、f804073、6ef37dd

---

## 8. 与 Jev 1.14 对比，以及 Decision Index

### 8.1 jev
- **目的**：在同一批题上把 Jev 1.14 和 Glance v1 配对比较。
- **代码入口**：
  - 构建请求：`python evals/jev/build.py <bench> --root ~/jev`
  - 调用模型：`python evals/jev/client.py --requests runs/<b>/requests.jsonl --out runs/<b>`，走 OpenRouter，**需要付费**
  - 打分：`python evals/jev/score.py all --root ~/jev --out summary.json`
- **数据**：Jev 的原始响应在 `Vision-Decision/jev-runs:runs/jev-1.14_runs_20261118.tgz`。打分不需要重新调用 API。
- **结果文件**：`outputs/jev/summary_all.json`、`outputs/jev/cost.json`、`outputs/jev/latency.json`
- **关键数字**（`summary_all.json`）：
  - gRefCOCO N-acc 两者都是 0.874，差值的 CI 为 [−0.007, +0.007]。
  - 把误拒答率卡在 1% 时，N-acc：Jev 0.725，Glance 0.845。
  - 总花费 $11.16（`cost.json`）。
- **状态**：完成。
- **提交**：98bff65、d49f501、b93ea73

### 8.2 di：Decision Index
- **代码入口**：按 `di/RUN.md` 里的步骤，依次执行 `bash di/colab_run.sh setup`、`bash di/colab_run.sh build_text` 等。RUN.md 开头写的「not launched」已经过时。
- **检查点**：`Vision-Decision/Glance-4B-preview:glance_config.json`，revision `b6237b4`，即 v1。
- **结果文件**：`di/results/text_scores.json`、`di/results/vision_summary.json`；逐行结果在 `Vision-Decision/di-runs:glance-v1/text-glance-v1/scores.json`。
- **关键数字**：文本榜 decision_index 41.26；视觉榜 public_vision_index 55.46。
- **状态**：完成。评测套件 kit 按条款不转存。
- **提交**：08ce8f7、28b8a02

---

## 9. Glance-Data / ground_v2

### 9.1 gv2-convert
- **目的**：把同组同学的 Glance-Data 转成 Glance 的记录格式，并用规则补上原数据缺的「单个物体上的方位词」正例。
- **代码入口**（在 gen-host 上运行）：
  - `python scripts/ground_v2/fetch_glance_data.py --out ~/gv2/glance_data`
  - `python scripts/ground_v2/convert_glance_data.py --gd ~/gv2/glance_data --ann ~/gv2/ann --out ~/gv2/gd_conv`
  - `python scripts/ground_v2/upload.py`
- **数据**：
  - 输入：`Vision-Decision/Glance-Data:data/diagnostic-00000-of-00001.parquet`
  - 产出：`Vision-Decision/glance-ground-v2:full/train-00000.jsonl`、`Vision-Decision/glance-ground-v2:diagnostic.jsonl`
- **关键数字**：full 视图共 51,177 条记录、1,894,336 道题（`scripts/ground_v2/README.md`）。
- **状态**：完成。
- **提交**：b63de64、04b5b17、bc1abd7、edc2935

### 9.2 gv2-diag
- **代码入口**：`bash scripts/ground_v2/vm_eval_diag.sh`，出表用 `scripts/ground_v2/diag_table.py`。
- **结果文件**：`outputs/ground_v2/diag/v1.json`、`outputs/ground_v2/diag/v1.1.json`；逐行输出在 `Vision-Decision/glance-ground-v2:eval/diag_2026-11-18/v1_rows.jsonl.gz`。
- **关键数字**（`v1.json`）：只给类别名的题 Acc@0.5 为 50.0；单个物体上的方位词题只有 26.3。
- **状态**：完成。
- **提交**：a97a790

### 9.3 gv2-pipeline（已废弃）
- **说明**：自建的三模块流水线（规则 → DeepSeek → 规则）试点。抽检通过 164/176，后来改用 Glance-Data。中间文件留在 gen-host 的 `~/gv2`。
- **提交**：b63de64

---

## 10. v1.1

### 10.1 v1.1
- **目的**：在 v1 基础上加高分辨率 GUI 数据，提高自然图像占比，并修正温度网格的 bug。
- **代码入口**：
  1. 下载：`scripts/v1/fetch.sh`，再跑 `scripts/v1/fetch_v11.sh`
  2. 构建题池：`scripts/v1/prep_v11.sh`
  3. 混合：`python -m dv.data.prep_v1 mix --out /content/data/prep_v1 --total N`
  4. 训练：`python train.py --cfg configs/v1.1_4b.json`
  5. 训练后评测和导出：`scripts/v1/post_train_v11.sh`
- **数据**：混合参数 N 没有记录下来，训练混合也只在已经回收的 VM dv-v1 上（见缺口）。v1.2 的 VM 上按同一流程重建过一份题池。
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:v1.1/ckpt_final.pt`；发布导出在 `Vision-Decision/glance-train-ckpts-legacy:v1.1/release/heads.safetensors`。
- **结果文件**：`outputs/v1.1/run_summary.json`、`outputs/v1.1/eval_results.json`、`outputs/v1.1/eval_results_gui.json`
- **关键数字**：
  - 训练规模：221,475 条记录，3,857 步（`run_summary.json`）。
  - gRefCOCO：N-acc 0.860，ECE 0.036（`eval_results.json`）。
  - ScreenSpot-Pro 在 8192 token 下的 46.8 已作废，以 7.4 节的 50.7 为准。
- **状态**：完成。
- **提交**：c007689、27979fb、0b8f9b4

---

## 11. v1.2 与多框（分支 `v12b-multibox`）

### 11.1 mb-pilot：多框读出 pilot
- **目的**：在 v1.1 上加多实例框读出，包括热图、集合匹配的逐框置信度和计数头。成功标准是多目标的集合 F1 明显上升，同时单目标精度和 N-acc 不下降。
- **代码入口**：
  - 构建数据：`python scripts/multibox/prep_pilot.py --ann data/ann --pool data/prep_v1 --prep data/prep --out data/prep_mb`
  - 训练和评测：`scripts/multibox/push.sh SESSION jointnb2`，它在 VM 上调用 `scripts/multibox/vm_run.sh`
  - 对比：`python scripts/multibox/compare.py`、`scripts/multibox/analyze_rows.py`
- **数据**：`Vision-Decision/glance-train-data:multibox_pilot/prep_mb/train_mb.jsonl`。这份数据之前只在本机，本次已上传。
- **检查点**：`Vision-Decision/glance-train-ckpts:multibox/v1.2_pilot/jointnb2/ckpt_final.pt`、`Vision-Decision/glance-train-ckpts:multibox/v1.2_pilot/jointnb/ckpt_final.pt`、`Vision-Decision/glance-train-ckpts-legacy:v1.2_pilot/heads/ckpt_final.pt`
- **结果文件**：`Vision-Decision/glance-train-ckpts:multibox/v1.2_pilot/jointnb2/eval/eval.json`；base / heads / jointnb 的结果在 `Vision-Decision/glance-train-ckpts-legacy:v1.2_pilot/`。git 里没有结果文件。
- **关键数字**（各 `eval/eval.json`）：

  | 指标 | base | jointnb2 阈值解码 | jointnb2 计数解码 |
  |---|---|---|---|
  | gRefCOCO val 多目标 F1 | 0.510 | 0.661 | 0.744 |
  | cocoall F1 | 0.243 | 0.561 | 0.639 |

  代价：8 个 split 的平均 Acc@0.5 从 0.928 降到 0.916。预注册标准第 (2) 条要求波动不超过 ±1，这一条没有通过。
- **状态**：完成。joint 那一组已废弃。
- **提交**：29c6fbb…de771e3（来自 `worktree-agent-a4403277b948a99e4`，已经合入 `v12b-multibox`）

### 11.2 v12-catfree：不含类别词的 grounding 数据
- **目的**：只用功能、泛称、关系来描述目标，构造正负例。答案由规则决定，措辞交给 VLM 写，再盲验一遍。
- **代码入口**：
  1. `python scripts/v12/catfree_scenes.py --ann ~/gv2/ann --gd ~/gv2/glance_data --n 24000 --out ~/v12/scenes.jsonl`
  2. `scripts/v12/vm_gen.sh`，生成模型是 Qwen3.8-27B-FP8
  3. `python scripts/v12/catfree_build.py --scenes scenes.jsonl,scenes_rel.jsonl --gen gen.jsonl,gen_rel.jsonl --out_dir out/`
- **数据**：
  - 产出：`Vision-Decision/glance-ground-v2:catfree_v12/catfree.jsonl`
  - 构建输入：`Vision-Decision/glance-train-data:v1.2_catfree/scenes.jsonl`。这份文件之前只在 gen-host 上，本次已上传。
- **结果文件**：`outputs/v1.2/spotcheck_catfree.md`
- **关键数字**：
  - 141,248 个正例、46,377 个负例，来自 20,899 张图（`catfree_v12/catfree_stats.json`）。
  - 人工抽检 53/56 正确（`spotcheck_catfree.md`）。
- **状态**：完成。
- **提交**：6c59e74、3b1e2cd、29c1afa

### 11.3 v12-train：v1.2 训练与预注册验收（进行中）
- **目的**：从 v1.1 起训，加入 Glance-Data 的 single 视图、catfree 数据和一致性损失。按预注册的 A–D 四组标准，只对最终检查点评一次。
- **代码入口**：VM 上执行 `bash scripts/v12/vm_chain.sh`，再用 `python scripts/v12/report.py --post <dir> --out outputs/v1.2/results.json` 生成报告。
- **检查点**：`Vision-Decision/glance-train-ckpts:v1.2/ckpt_final.pt`。训练已经完成。
- **结果落点**：评测结果会推到 `Vision-Decision/glance-train-ckpts:v1.2/post/`；验收表写到 `outputs/v1.2/results.json`；预注册标准见 `outputs/v1.2/PREREG.md`。
- **状态**：**进行中**，VM `v12` 上正在跑评测。
- **提交**：71cbc87…7bc7624（`v12b-multibox`）。另有 2476f2b 只在 `worktree-agent-ab225d57487eeae40` 上，合并时需要一起带上。

### 11.4 v12b：v1.2 加多框（进行中）
- **代码入口**：`bash scripts/v12b/vm_v12b.sh`，内部依次是 `scripts/v12b/build_mix_b.py` → `train.py --cfg configs/v1.2b_heads.json` → `train.py --cfg configs/v1.2b_joint.json` → 评测。
- **结果落点**：`Vision-Decision/glance-train-ckpts:v1.2b/`；预注册标准见 `outputs/v1.2b/PREREG.md`。
- **状态**：**进行中**，排在 v1.2 评测之后。
- **提交**：a68fda5、ec73a7f、0178398

---

## 12. scan：一个问题扫多张图

### 12.1 scan-engine
- **目的**：每张图或每个片段的前缀状态只算一次并缓存。来了新问题，一次前向扫完 N 个单元，每个单元的答案和单独询问时相同。
- **代码入口**：
  - 引擎：`scan/engine.py`、`scan/api.py`
  - 精确性测试：`python -m scan.test_scan_equivalence`、`python -m scan.bench_scan --parts equiv,fresh`
- **检查点**：`Vision-Decision/glance-train-ckpts-legacy:v1/ckpt_final.pt`
- **结果文件**：`outputs/scan/phase1_tables.md`
- **关键数字**：GPU bf16 下，所有设置的决策一致率都是 1.000（`phase1_tables.md` 的 Exactness 表）。
- **状态**：完成。
- **提交**：5932aef、2860fd0、49cc3ec、4bc4f0a

### 12.2 scan-phase1：延迟
- **代码入口**：
  - `bash scan/vm/setup.sh --vllm`
  - `python -m scan.bench_scan --ckpt /content/ckpt/v1/ckpt_final.pt --parts clips,cached,fresh,qfirst,acc,fp8`
  - `scan/vm/chain1.sh`
  - 出表：`python3 scan/phase1_tables.py outputs/scan/raw/scan-a`
- **结果文件**：`outputs/scan/raw/scan-a/bench_g4.json`、`outputs/scan/phase1_tables.md`
- **关键数字**（`phase1_tables.md`）：
  - 缓存好的 64 个片段：Glance 63.6 ms，vLLM bf16 343 ms。
  - 全新图片上 Glance 没有优势：每张 256 token、32 张图时，Glance 482 ms，vLLM 236 ms。
- **状态**：完成。
- **提交**：5d73176、0fb9056、3982d9b

### 12.5 scan-short-v1：短读出块蒸馏
- **目的**：把 yes/no 读出块从 26 个 token 缩到 14 个，让扫 256 个片段的耗时进入 100 ms 以内。
- **代码入口**：
  - 准备：`python3 scan/distill_prep.py --root ~/scan/charades --out ~/scan/distill`
  - 教师打标：`python -m scan.distill_teacher`
  - 训练：`train.py --cfg configs/scan_short_v1.json`
  - 以上串在 `scan/vm/chain4.sh` 里；对齐检查用 `scan/vm/chain7.sh`
- **数据**：
  - 训练片段：`Vision-Decision/scan-pilot-data:charades_train_clips_2fps4.tar`
  - 片段清单：`Vision-Decision/glance-train-data:scan_distill/records.jsonl`
- **检查点**：`Vision-Decision/glance-train-ckpts:scan/short_v1/ckpt_final.pt`
- **结果文件**：私有研究仓库中的 `results/pilot_clips_short_analysis.json`、glance-engine 的 `outputs/scan/raw/scan-a/latency_short.json`
- **关键数字**：
  - 256 个片段 97.3 ms（`latency_short.json`）。
  - 短块和长块的 AUROC 都是 0.883。
  - 代价：框 Acc@0.5 从 0.886 降到 0.858（`parity150_short.json` 对比 `parity150_v1.json`）。
- **状态**：完成。
- **提交**：41d9c3c、b5b8109、24f2bb5、4819469、6d3b9a0

### 12.6 scan-gv：Glance-Video 片段 pilot 和 short_gv0
- **代码入口**：
  - main 上：`GV=glance-video-v0/ckpt_last.pt TAG=gvlast bash scan/vm/chain6.sh`
  - `glance-video` 分支上：`scan/vm/chain9_gvdistill.sh`
- **检查点**：
  - 输入：`Vision-Decision/glance-train-ckpts:glance-video-v0/ckpt_final.pt`
  - 产出：`Vision-Decision/glance-train-ckpts:scan/short_gv0/ckpt_final.pt`
- **结果文件**：私有研究仓库中的 `results/pilot_clips_gvlast_analysis.json`；gvfinal 和 short_gv0 的结果在 `Vision-Decision/glance-train-ckpts:results/gv0/scan/pilot_clips_gvfinal_analysis.json`，对应的仓库文件只在 `glance-video` 分支上。
- **关键数字**：
  - gvlast 的片段 AUROC 为 0.894（`pilot_clips_gvlast_analysis.json`）。
  - short_gv0 的短块和长块 AUROC 都是 0.891。
- **状态**：完成。
- **提交**：1505d6e、e1f8bc1、8c1b5e2（main）；c2bef8c（`glance-video`）

---

## 13. Glance-Video（分支 `glance-video`）

### 13.1 gv-data：片段池与去污染
- **目的**：用 K700 val、Countix、Charades/STA、RepCount-A 构建片段池和类型化问题。针对 Video-Index 做四层去污染：数据集级、id 级、SHA-256、1 fps 的 pHash。
- **代码入口**：`glance-video` 分支的 `scripts/video/` 下（`prep_clips.py`、`gen_records.py`、`decon_list.py`、`push_shards.py`），在 gen-host 的 `~/gv` 下运行。
- **数据**：`Vision-Decision/glance-video-v0:shards/manifest.json`、`Vision-Decision/glance-video-v0:records_full/summary.json`、`Vision-Decision/glance-video-v0:meta/decon/exclusions.json`
- **关键数字**（`records_full/summary.json`）：train_k700 共 62,809 条记录，train_charades 共 294,539 道题。
- **状态**：完成。
- **提交**：87dbca8、e83fbd5

### 13.2 gv-v0：训练与离线评测
- **代码入口**：
  - 训练：`MODE=full bash scripts/video/vm_train.sh`，内部是 `train.py --cfg configs/video_v0.json`，从 v1.1 初始化
  - 评测：`python scripts/video/eval_video.py typed|sta|repcount|tb|vi --out X.json`
- **检查点**：`Vision-Decision/glance-train-ckpts:glance-video-v0/ckpt_final.pt`。500 步试训（gv_a）的结果在 `Vision-Decision/glance-video-v0:results/gv_a/pilot_typed.json`。
- **结果文件**：`Vision-Decision/glance-train-ckpts:results/gv0/full_typed.json`，同目录下还有 `*_repcount.json`、`*_textheldout.json`、`full_vi_v1_1.json`。
- **关键数字**（本次重新从 HF 读取核对）：
  - Countix 计数准确率：0.929，底座 0.294。
  - RepCount-A 的 argmax MAE：0.192，底座 0.524。
  - 留出文本集 3,000 条，没有出现遗忘。
- **状态**：完成。
- **提交**：e83fbd5、e46baec

### 13.3 gv-tool：ClipBackend 与 vdmscan
- **代码入口**：
  - 扫描服务：`video/vdm/gv/scan_server.py`
  - 命令行工具：`video/vdm/gv/vdmscan`
  - 给规划器看的工具说明：`video/vdm/gv/TOOL_scan.md`
- **状态**：完成，Video-Index 的配对实验正在使用。
- **提交**：839e734、b4ba6d4、72f9a6b

---

## 14. Video-Index

Video-Index 的视频和帧只能用于评测：不能用来训练、标定或调阈值，也不能转存（见 `video/design.md`）。下载脚本是 `video/sandbox/fetch.sh`。

### 14.2 vi-sandbox：本地复刻官方协议
- **代码入口**（在 gen-host 的 `~/videoindex` 下）：
  1. `docker build -t vi-agent:cc2.1.300 sandbox/`
  2. `python3 sandbox/run.py --start-gateway`
  3. `python3 sandbox/run.py --run full1 --vdm`
- **结果文件**：`video/results/SNAPSHOT.md`、`video/results/DEVIATIONS.md`
- **关键数字**（`SNAPSHOT.md`）：
  - DeepSeek 不带工具：11/66。
  - Gemini 加上 vdm v0 后，正确率变化 Δ = −16.7 个百分点，95% CI [−33.3, −4.2]。
- **状态**：完成。vdm v0 是负收益。
- **提交**：287e30d、f93502b、d4a1b44

### 14.3 vi-choose：只用 choose 的下界
- **代码入口**：`video/vdm/choose_v1_0.py`
- **结果文件**：`video/results/SNAPSHOT.md`（glance-video）；逐题结果 `choose_only.jsonl` 含答案键，已移出 git，在组织内的私有数据集里（`glance-video/video/vdm/results/`）
- **关键数字**：在 v1.1 上，取 argmax 只对 2.1%，取 argmin 反而对 66.7%，随机猜是 24.8%。
- **状态**：完成。
- **提交**：f93502b、d4a1b44

### 14.4 vi-gate：vdmscan 离线门槛
- **代码入口**：`glance-video` 分支上的 `python -m scan.vi_gate`，以及 `scan/vi_gate_e2e.py`。
- **结果文件**：`video/results/vi_scan/gate_stats.json`。这个文件属于还在跑的配对实验，目前没有提交。
- **关键数字**：
  - 留出片段 AUROC 为 0.899。
  - `when` 的 FDR 为 0.300，目标是 0.1，没有达标。
- **状态**：完成。门槛没有全部通过，但仍决定上 Video-Index，已记为偏差。
- **提交**：5c22442

### 14.5 vi-scan-paired：DeepSeek 不带工具与加 vdmscan 对比（进行中）
- **代码入口**：
  - 在 gen-host 上：`~/videoindex/run_scan_exp.sh`，这个脚本还没进仓库
  - 统计：`python3 video/sandbox/paired.py --a ds_base_hard --b ds_scan_hard --ref full1`（`glance-video` 分支）
- **检查点**：short_gv0，扫描服务跑在 VM `dv-scan` 上。
- **结果落点**：
  - 中间轨迹在 gen-host 的 `~/videoindex/data/runs/`，不转存。
  - 最终结论写进 `video/results/`，包括 `DEVIATIONS.md` 和 `vi_scan/`，提交到 `glance-video` 分支。
- **状态**：**进行中**。中间计数不要引用。
- **提交**：2516782、72f9a6b

---

## 15. 博客

### 15.3 blog / slides
- **博客**：
  - 生成 BBB 帧数据：`python3 outputs/blog/run_glance.py <bbb.mp4> <frame_dir> glance_bbb.json`
  - 生成插图：`python3 -I outputs/blog/make_post_figs.py`
  - 素材在 glance-video 的 `outputs/blog/`。`coverage.json`、`codex_trace.json` 和 `trace_scripts/` 摘录了 Video-Index 的题目和 agent 轨迹，2026-11-20 已从 git 历史删除，存于组织内的私有数据集（`glance-video/outputs/blog/`）。博客正文在另一个仓库。
- **幻灯片**：含 Video-Index 题目细节，2026-11-20 已从 glance-video 历史删除，存于组织内的私有数据集（`glance-video/slides/`）（取回后 `cd slides && pnpm install && pnpm export`）。
- **提交**：d45cc7b（博客素材）；abb97c6、4efe2bf（幻灯片）

---

## 16. 分支

| 分支 | 内容 | 建议 |
|---|---|---|
| `v12b-multibox` | v1.2 的 catfree 数据、训练链和预注册标准；多框模型和评测；v12b 配方（领先 main 40 个提交） | 等 v1.2 和 v12b 结果提交后合并，同时 cherry-pick 2476f2b |
| `worktree-agent-ab225d57487eeae40` | v1.2 训练链。比 `v12b-multibox` 只多 2476f2b 一个提交（检查点改成逐个文件上传） | 只需要 cherry-pick 这一个提交 |
| `worktree-agent-a4403277b948a99e4` | 多框 pilot，已经全部并入 `v12b-multibox` | 不用单独合并 |
| `glance-video` | Glance-Video 的数据、训练和评测；ClipBackend；vdmscan；Video-Index 离线门槛；配对实验脚本（领先 9 个提交） | 等配对实验结束、`DEVIATIONS.md` 和 `vi_scan/` 提交后合并 |
| `worktree-agent-ad2185ba3ae6bc7f3` | sprint_metrics，内容和 main 上的 08503e3 相同 | 不用合并 |
| `repro-index` | 本索引，以及之前没入库的结果文件 | 等在跑的作业结束后合并 |

其余 `worktree-agent-*` 分支都已经合进 main。

---

## 17. 缺口

下面这些东西目前不在 GitHub 或 HF 上，或者没法逐字节复现。路径和 HF 文件是否存在由 `scripts/check_experiments_index.py` 检查，它的输出见本节末尾。

**需要尽快处理（VM 一回收就会丢）：**

1. **v1.2 训练混合 `/content/data/prep_v12/train.jsonl`（473 MB）和 v1.1 流程重建的题池 `/content/data/prep_v1/pool_*.jsonl`（约 600 MB）只在 VM `v12` 上。** 那台 VM 正在跑评测，这次没有从它上面拉大文件。v1.2 的 agent 收尾时应该把这两份推到 `Vision-Decision/glance-train-data:v1.2/`。同一台 VM 上的 `v12_chain2` 链脚本也还没进 git，而 `scripts/v12b/vm_v12b.sh` 依赖它的日志。
2. **Video-Index 配对实验的辅助脚本**（`~/videoindex/run_scan_exp.sh`、`scan_health.sh`、`vdmerr.py`）、门槛校准文件 `~/videoindex/gate/CALIB.json` 和 Docker 镜像都只在 gen-host 上。实验还在跑，结束时应该把它们提交到 `glance-video` 分支。

**已经无法逐字节复现（来源已不存在）：**

3. **v1 的原始训练混合**：VM 已回收。HF 上的 `genv1_prep.tgz` 是用同一提交重建的，但不保证逐条相同。
4. **v1.1 的训练混合**：VM dv-v1 已回收，`mix --total` 的参数也没有记下来，只能按 10.1 节的流程重建。
5. **S1 GUI 的 ft2 检查点和 GUI 续训混合**（`prep_gui_v2`）：上传时个人 HF 空间已满，推送失败。属于负结果，影响小。
6. **hnft / hnft_mt 的训练混合**：可以用 `make_hnft.py` 和 `make_mt.py` 重新生成。
7. **ablation-v2 的种子扩展**：`_s2` 和 `neg05_s1` 没有跑完，没有检查点。
8. **ablation-v3 的 IoU 头权重**：只保存了分数（`head_scores.npz`）和特征。
9. **equiv-toy 的输出**、**scan 的 kernel A/B 结果**（`kernel_*.json`）：从来没有存档。

**文档或命名上的问题：**

10. **27B 的评测命令**没有写进仓库。
11. **Glance-Video 的 VM 评测链脚本**和 TemporalBench 的下载脚本不在仓库里。
12. **gen_v1 的逐行输出和分析**：还在产出中。
13. **原个人仓库已搬到组织**：v0、v1、v1.1、ablation、sprint_robust 等的检查点现在在 `Vision-Decision/glance-train-ckpts-legacy`（已完成）。
14. **HF id 已统一**：脚本和文档里的 guisyn-v0、glance-robust-ref、glance-video-v0、Glance-4B-preview 都已改用 `Vision-Decision/` 下的 id（已完成）。结果 JSON 和日志里保留运行时记录的旧 id，旧 id 会重定向。
15. **按条款不转存的数据**：Decision Index 的 kit（`data/di/`）、Video-Index 的视频和帧。只记录了下载方式。

**检查脚本输出：**

<!-- CHECK-OUTPUT -->

运行方式：`python scripts/check_experiments_index.py`。脚本会把本文引用的仓库路径拿去和 main 及三个未合并分支的 git 树比对，把 HF 引用拿去查 `list_repo_files`。HF 检查需要一个对私有仓库有读权限的 token，加 `--no-hf` 可跳过这一步。

最近一次运行（2026-11-19）共检查了 249 个仓库路径和 58 个 HF 引用。结果分三类：

- **只在未合并分支上的路径**：25 个，属于预期，合并后即在 main 上：
  - `v12b-multibox`：`scripts/multibox/`、`scripts/v12/`、`scripts/v12b/`、`configs/v1.2b_*.json`、`outputs/v1.2*/`
  - `glance-video`：`scripts/video/`、`configs/video_v0.json`、`scan/vi_gate_e2e.py`、`scan/vm/chain9_gvdistill.sh`、`video/sandbox/paired.py`、`video/vdm/gv/TOOL_scan.md`
- **不存在的路径**：11 个，没有一个是错误引用：
  - **命令参数，不是仓库文件**：`data/ann`、`data/prep_mb`、`data/prep_v1`。它们是 `prep_pilot.py` 命令里的本地目录。
  - **按条款不入库**：`data/di/`。
  - **进行中实验的结果落点，还没生成**：`outputs/gen_v1/`、`outputs/v1.2/results.json`、`video/results/vi_scan/gate_stats.json`、`Vision-Decision/glance-train-ckpts:v1.2/post/`、`Vision-Decision/glance-train-ckpts:v1.2b/`。
  - **建议的上传位置（见缺口第 1 条）**：`Vision-Decision/glance-train-data:v1.2/`。
