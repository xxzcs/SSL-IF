# 项目进度与交接记录

> 使用规则：后续继续本项目时，先阅读本文件，再检查最新日志与完成标记。每次改变实验计划、代码、配置或得到新结果后更新本文件。不要仅凭 `*_DONE` 判断成功，必须同时检查日志、checkpoint 和汇总 CSV。

最后更新：2026-07-21 14:36 CST

## 当前目标

在 BUS 半监督二分类任务上验证 IF/IFCF 是否能稳定融合到不同 SSL Base。正式比较统一使用固定训练终点的 `latest_model.pth`，避免根据测试表现选择 `model_best.pth`。后续五折交叉验证也必须使用固定训练长度，测试折不能参与 checkpoint 或超参数选择。

共同 BUS 设置：

- 标注样本 878，未标注样本 3512，二分类。
- ResNet-18，50 epoch / 5500 iterations，batch size 8，uratio 30。
- 数据划分：`../data_split/28/labeled_images_20_9.pth` 和 `unlabeled_images_80_9.pth`。
- 当前 FixMatch/ReFixMatch 公平比较阈值：`p_cutoff=0.9`。

## 当前正在运行的任务

任务：修正后的 ReFixMatch Base，BUS，5 seeds，最终只汇总 latest checkpoint。

- 启动脚本：`rerun_refixmatch_fixed_base.sh`
- 主日志：`logs/refixmatch_fixed_base_20260721_121817.log`
- 模型目录：`saved_models/usb_cv/refixmatch_fixed_base_s{1..5}`
- 预期汇总：`results/refixmatch_fixed_base_summary.csv`
- 完成标记：`REFIXMATCH_FIXED_BASE_DONE`
- GPU：RTX 3090；conda 环境：`wssl`。

截至 2026-07-21 14:36：

- seed 1：完成。
- seed 2：完成。
- seed 3：完成。
- seed 4：完成。
- seed 5：已于 14:35 启动，尚未产生第一个正式 checkpoint。
- 最终 5-seed latest 汇总尚未生成，完成标记尚不存在。

当前训练由 Codex 的外部 GPU 执行会话启动。若会话丢失，先检查进程和日志；脚本会跳过已有 `latest_model.pth`，但注意：训练中途也会写 latest，因此不能仅凭 latest 文件判断一个 seed 已完整跑完。应检查该 seed 的 `log.txt` 是否包含 `GPU 0 training is FINISHED` 或 5500 iteration。

## ReFixMatch 问题与修复

旧实现存在实质性错误：

1. 文件注释错误地把 FixMatch 论文 `arXiv:2001.07685` 当成 ReFixMatch。
2. hard CE 与 soft KL 实际作用于同一个高置信度 mask。
3. 低置信度样本仍被丢弃，违背 ReFixMatch 的核心设计。
4. KL 路径把已经 softmax 的概率再次当 logits 做 softmax，并且错误地对 strong logits 同时做温度缩放。

正确实现依据 ReFixMatch 论文 `arXiv:2308.07509` 的公式 3--6：

- `max(q) >= tau`：hard pseudo-label CE。
- `max(q) < tau`：以 `softmax(weak_logits / T)` 为目标计算 KL。
- 两部分 mask 互补，损失均按整个无标注 batch 求均值。

修正后冒烟日志已验证：高置信度比例为 0 时，`unsup_loss=0`，但 `unsup_loss_refix>0` 且 `low_conf_ratio=1`，说明低置信度分支确实工作。

## 本轮修改文件及原因

以下只记录本轮 Codex 明确进行的修改；仓库中其他大量未提交修改属于此前工作，不能擅自覆盖或清理。

### `semilearn/algorithms/refixmatch/refixmatch.py`

- 修正论文链接。
- 新增 `compute_refixmatch_unsup_loss`，实现互补的高置信度 hard CE / 低置信度 sharpened soft KL。
- 高置信度伪标签固定为 hard label，符合论文。
- 日志增加 `unsup_loss_refix` 和 `low_conf_ratio`，便于诊断。

### `semilearn/algorithms/refixmatch/refixmatch_ifcf.py`

- ReFixMatch+IFCF 复用修正后的 ReFixMatch 无监督损失，避免复制错误公式。
- 日志同样增加 soft KL 和低置信度比例。

### `config/usb_cv/refixmatch/refixmatch_bus_878_0.yaml`

- 修正文档说明和 `load_path`。
- 最初按原论文默认改为 0.95；为与当前 BUS FixMatch 严格公平比较，最终设为 `p_cutoff=0.9`。

### `tests/test_refixmatch_losses.py`

- 新增损失测试：验证高/低置信度 mask 互补、低置信度 KL 公式正确、全高置信度时 soft loss 为零。
- 2026-07-21 已通过 2 个测试。

### `rerun_refixmatch_fixed_base.sh`

- 新增修正版 ReFixMatch Base 的 GPU 冒烟和 5-seed 训练流水线。
- 使用新实验名，避免覆盖旧的错误实现结果。
- 最终仅汇总 `latest_model.pth`。

### `tools/build_bus_experiment_table.py`

- 收集 BUS 结果 CSV 和模型日志参数，生成完整实验清单。

### `results/BUS_ALL_EXPERIMENTS.md`、`results/BUS_ALL_EXPERIMENTS.csv`

- 自动生成的 BUS 历史实验总表，共 236 条评估记录；同时包含 best/latest，仅用于盘点，正式论文结果应筛选 latest。

## 旧 ReFixMatch 结果的处理

旧的 `refixmatch_base_s*`、`refixmatch_ifcf_hard_s*`、`refixmatch_ifcf_soft_s*` 来自错误实现，不应进入正式论文比较，但目前没有删除。

曾启动的 p=0.95 修正版 seed 1 在 220 iterations 时停止，checkpoint 已保留在：

`saved_models/usb_cv/archived_refixmatch_p095/refixmatch_fixed_base_s1_interrupted_20260721_1217`

不要把该中断实验混入 p=0.9 的正式结果。

## 已验证的重要结果口径

按旧实验的 latest AUC，IF 在各框架上的平均 AUC 实际均高于 Base，但部分 ACC/Sen/Spe/F1 有权衡。不能声称“五项指标全部普遍提升”，更稳妥的主张是 IF 改善判别/排序能力，最终仍需五折和配对显著性检验支持。

- FreeMatch：0.840444 -> 0.844669。
- SoftMatch：0.838439 -> 0.845493。
- AdaMatch：0.841454 -> 0.847038。
- FlexMatch：0.837887 -> 0.842355。
- 旧错误 ReFixMatch：0.795326 -> 0.813835（该组不能作为正式证据）。

## 后续任务（按顺序）

1. 等当前 ReFixMatch fixed Base seed 5 完成，并检查：
   - 五个 seed 均实际到达 5500 iterations；
   - 日志无 Traceback、CUDA OOM、NaN；
   - `REFIXMATCH_FIXED_BASE_DONE` 存在；
   - `results/refixmatch_fixed_base_summary.csv` 是 latest 的五种子均值±标准差。
2. 将修正后的 ReFixMatch Base 与 FixMatch Base（同为 `p_cutoff=0.9`）比较，确认性能进入合理区间。
3. Base 正常后，再新建独立实验名运行 ReFixMatch fixed + IFCF hard，5 seeds，latest。
4. 是否运行 ReFixMatch fixed + IFCF soft，视 hard 结果决定；不要直接复用旧 checkpoint。
5. 对 FreeMatch/FlexMatch 的 IF 权重只在独立验证集选择，建议候选 `0.1/0.25/0.5/1.0`，不能根据测试集反复挑参。
6. 最终五折交叉验证：每折固定训练长度并报告 latest，计算五项指标均值±标准差及 Base-vs-IF 配对差值/显著性。

## 安全与复现提醒

- 不删除或覆盖用户已有未提交修改。
- 不把 `model_best.pth` 用作正式主表结果。
- 不因看到测试指标而改变参数；参数必须由训练/验证数据决定。
- ReFixMatch 旧结果与修正版结果必须使用不同实验名。
- 普通 Codex 沙箱看不到 `/dev/nvidia*`；GPU 训练需获准在沙箱外运行，但仍使用同一个 `wssl` conda 环境。
