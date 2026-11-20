# Exp 7b：扫描结果用集合级统计量代替 max（免训练）——收益很小

**结论：** 在缓存的 Glance-Video 片段扫描上，把"整段视频里有没有"的统计量从 max 换成 top-2 均值，AUROC 提高 **+0.29** [+0.21, +0.38]。在 10% 误报下，漏检率从 0.199 降到 0.189。集合级统计量能用，但收益只有零点几个点，不能作为卖点。真正有用的是选片时的时间 NMS：top-3 中至少一个命中的比例从 0.882 升到 **0.930**，代价是 precision@3 从 0.70 降到 0.52。这正是给规划器挑片段时需要的权衡。

- 数据：Charades-STA test，2 fps、4 帧一段、步长 2 帧。共 12,475 个扫描（查询 × 视频），其中 1,394 个正例。分数是 Glance-Video final 的逐片段 yes-logit（`Vision-Decision/glance-train-ckpts:results/gv0/scan/pilot_clips_gvfinal.json.gz`）。
- 校准：任何统计量都可以用 max-stat 式的保形分位数控制误报。按查询对半切分校准 / 测试，重复 50 次。
- CI：按查询做 bootstrap。代码：`arch/scan/setsel.py`；结果：`setsel.json`。

| 统计量 | 扫描 AUROC | 相对 max 的 Δ（95% CI） | 漏检率 @ 误报 0.1 |
|---|---|---|---|
| max | 93.23 | — | 0.199 |
| top-2 均值 | **93.52** | +0.29 [+0.21, +0.38] | **0.189** |
| top-3 均值 | 93.46 | +0.23 | 0.194 |
| 3 段平滑后取 max | 93.43 | +0.20 | 0.200 |
| logsumexp | 91.90 | −1.33 | 0.240 |
| noisy-OR | 89.17 | −4.06 | 0.348 |
| 均值 | 84.17 | −9.06 | 0.500 |

正例扫描内的选片：

| 选法 | hit@1 | 任一命中@3 | precision@3 |
|---|---|---|---|
| 独立 top-k | 0.750 | 0.882 | 0.696 |
| 平滑后 top-k | 0.758 | 0.852 | 0.739 |
| 时间 NMS（±2 段） | 0.750 | **0.930** | 0.519 |

时间方向（TDS，正放与倒放对比）的诊断需要 GPU，另做（exp 7a）。

# Exp 7a: does a cached clip carry the arrow of time? (TDS, zero training)

**Bottom line.** Yes, partly. Each positive clip is encoded twice, forward and with its frames reversed, and the Glance-Video final checkpoint is asked the same question both times. For direction-paired actions (open/close, sit down/stand up, take/put, …) the yes-logit moves by |Δ| = **0.90** [0.85, 0.95] on average. For non-directional actions it moves by **0.48** [0.45, 0.51], which is the baseline sensitivity to frame order. Time-direction accuracy, i.e. the fraction of clips where the forward logit beats the reversed one, is **0.60** [0.58, 0.62]. Paired accuracy, where the query must also beat its opposite action more under forward than under reversed playback, is **0.63** [0.61, 0.65]. Both are above chance but weak.

**Implication.** TDS is clearly non-zero, so the direction signal is already in the cached clip prefix; repairing AoT does not require touching the encoder side. Cheap readout-level training, using reversed clips as hard negatives for directional queries (ArrowRL-style, but supervised), is a credible phase-2 item. Retraining the video encoder is not needed first. A cross-clip set reranker will not fix within-clip direction, so this is a separate item.

- n = 659 directional scans (1,962 clips) and 541 non-directional scans (1,612 clips). Up to 3 clips are sampled inside the span of each positive Charades-STA test scan. Clips are 4 frames at 2 fps, 96 visual tokens per 2-frame group, exactly as in the scan.
- Code: `arch/scan/tds.py`. Rows: `Vision-Decision/glance-train-data:arch/scan_tds/tds_gv0.json`. Summary: `tds_summary.json`.
