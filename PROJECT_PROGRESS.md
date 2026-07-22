# 项目进度与交接记录

> 使用规则：后续继续本项目时，先阅读本文件，再检查最新日志与完成标记。每次改变实验计划、代码、配置或得到新结果后更新本文件。不要仅凭 `*_DONE` 判断成功，必须同时检查日志、checkpoint 和汇总 CSV。

最后更新：2026-07-22 11:23 CST

## 当前目标

在 BUS 半监督二分类任务上验证 IF/IFCF 是否能稳定融合到不同 SSL Base。正式比较统一使用固定训练终点的 `latest_model.pth`，避免根据测试表现选择 `model_best.pth`。后续五折交叉验证也必须使用固定训练长度，测试折不能参与 checkpoint 或超参数选择。

共同 BUS 设置：

- 标注样本 878，未标注样本 3512，二分类。
- ResNet-18，50 epoch / 5500 iterations，batch size 8，uratio 30。
- 数据划分：`../data_split/28/labeled_images_20_9.pth` 和 `unlabeled_images_80_9.pth`。
- 当前 FixMatch/ReFixMatch 公平比较阈值：`p_cutoff=0.9`。

## 当前正在运行的任务

任务：夜间自动队列已完成；现已启动 SimMatch hard-IF 统一配方 3seed screen。

- 脚本：`run_simmatch_hard_if_unified_screen.sh`
- 日志：`logs/simmatch_hard_if_unified_screen_20260722.log`
- 模型目录：`saved_models/usb_cv/simmatch_if_hard_w1_unified_s{1..3}`
- 预期汇总：
  - `results/simmatch_hard_if_unified_3seed_val.csv`
  - `results/simmatch_hard_if_unified_3seed_test.csv`
- 完成标记：`SIMMATCH_HARD_IF_UNIFIED_3SEED_DONE`
- 配置：SimMatch `use_da=True`、`p_cutoff=0.9`、`T=0.1`、`lr=0.0046875`、`ema_m=0.999`；IF 为 `if_target=hard`、`ifrank_combine=multiply_balanced`、`ifrank_loss_weight=1.0`、`if_lambda=1`、`csim_lambda=1`、`corrT=0.9`、`num_references=4`、`ref_select=by_instance`、`ref_cand_k=8`、`use_strong_if=True`、`ifrank_warmup_epochs=5`、`ifrank_warmup_mode=zero`。

为支持该统一配方，`semilearn/algorithms/simmatch_if/simmatch_if.py` 新增了 SimMatchIF 的 `ifrank_warmup_epochs` / `ifrank_warmup_mode` 参数；默认仍为历史兼容的 `1/zero`，新实验显式设置为 `5/zero`。

- 总控脚本：`overnight_auto_if_queue_20260721.sh`
- 总控日志：`logs/overnight_auto_if_queue_20260721.log`
- 总控完成标记：`OVERNIGHT_AUTO_IF_QUEUE_20260721_DONE` 已生成。
- 原独立 FreeMatch 等待脚本 `queue_freematch_warmup1_complete5.sh` 已由总控脚本替代，避免在 Flex 3seed 完成后和补 seed4/5 抢 GPU。

夜间自动顺序与结果：

1. 等待当前 FlexMatch 3seed 屏幕完成。
2. 将 FlexMatch `lr=0.0046875 / p=0.9 / hard-IF weight1 / warmup5` 从 3seed 补到 5seed，并生成：
   - `results/flexmatch_lr0046875_w1_5seed_val.csv`
   - `results/flexmatch_lr0046875_w1_5seed_test.csv`
   - `FLEXMATCH_LR0046875_W1_5SEED_DONE`
   - Test latest：Base AUC/ACC/Sen/Spe/F1 = 0.840245/0.781557/0.743624/0.820862/0.776013；IF = 0.846501/0.787022/0.753020/0.822253/0.782548。差值 +0.006256/+0.005465/+0.009396/+0.001391/+0.006535。
   - Val latest：Base AUC/ACC/Sen/Spe/F1 = 0.856872/0.784426/0.721342/0.849791/0.772989；IF = 0.866277/0.792350/0.734228/0.852573/0.782556。差值 +0.009405/+0.007924/+0.012886/+0.002782/+0.009567。
3. 补 FreeMatch 早期好配置 `soft-IF / warmup1 / weight1 / lr=0.0046875 / ema=0` 的 seed4/5，并生成：
   - `results/freematch_warmup1_5seed_summary.csv`
   - `results/freematch_warmup1_5seed_val.csv`
   - `FREEMATCH_WARMUP1_5SEED_DONE`
   - Test latest：Base AUC/ACC/Sen/Spe/F1 = 0.839428/0.782377/0.741477/0.824757/0.776145；IF = 0.846625/0.787022/0.750067/0.825313/0.781828。差值 +0.007197/+0.004645/+0.008590/+0.000556/+0.005683。
   - Val latest：Base AUC/ACC/Sen/Spe/F1 = 0.854405/0.785246/0.719463/0.853408/0.773211；IF = 0.865018/0.791530/0.732349/0.852851/0.781451。差值 +0.010613/+0.006284/+0.012886/-0.000557/+0.008240。
4. 跑修正版 ReFixMatch `p=0.9 / hard-IF weight1 / warmup5` 5seed，并生成：
   - `results/refixmatch_fixed_ifw1_summary.csv`
   - `results/refixmatch_fixed_ifw1_val.csv`
   - `REFIXMATCH_FIXED_IFW1_5SEED_DONE`
   - Test latest：Base AUC/ACC/Sen/Spe/F1 = 0.838080/0.776639/0.720000/0.835327/0.766375；IF = 0.843840/0.782787/0.739866/0.827260/0.776107。差值 +0.005760/+0.006148/+0.019866/-0.008067/+0.009732。
   - Val latest：Base AUC/ACC/Sen/Spe/F1 = 0.852802/0.780464/0.699060/0.864812/0.764183；IF = 0.861377/0.789617/0.719195/0.862587/0.776751。差值 +0.008575/+0.009153/+0.020135/-0.002225/+0.012568。
5. 汇总已有 SimMatch latest checkpoints，不新开未锁定的 SimMatch 大规模搜索：
   - `results/simmatch_existing_latest_test.csv`
   - `results/simmatch_existing_latest_val.csv`
   - `SIMMATCH_EXISTING_LATEST_SUMMARY_DONE`
   - Test latest：SimMatch base AUC/ACC/Sen/Spe/F1 = 0.865228/0.797131/0.771544/0.823644/0.794549；现有 IF-f5-c4 = 0.864768/0.798497/0.784430/0.813074/0.798360。差值 -0.000460/+0.001366/+0.012886/-0.010570/+0.003811。
   - Val latest：SimMatch base AUC/ACC/Sen/Spe/F1 = 0.884286/0.799727/0.747651/0.853686/0.791658；现有 IF-f5-c4 = 0.889217/0.806011/0.764564/0.848957/0.800332。差值 +0.004931/+0.006284/+0.016913/-0.004729/+0.008674。

当前 FlexMatch 3seed 子任务细节：

- 启动脚本：`run_flexmatch_lr0046875_w1_screen.sh`
- 主日志：`logs/flexmatch_lr0046875_w1_screen_20260721.log`
- 模型目录：`saved_models/usb_cv/flex_lr0046875_{base,ifw1}_s{1..3}`
- 预期验证汇总：`results/flexmatch_lr0046875_w1_val_screen.csv`
- 完成标记：`FLEXMATCH_LR0046875_W1_SCREEN_DONE`
- 配置：p=0.9、IF为hard/weight1/zero5/strong、`T=0.5`、`corrT=0.9`、`lr=0.0046875`。

注意：基础 FreeMatch/FlexMatch YAML 当前写有 `lr=0.009375`，但共同BUS口径及早期泛化脚本明确使用 `0.0046875`。2026-07-21 发现首轮权重筛选误继承0.009375后已立即停止；随后又确认在新学习率下应先重建Base/weight1基准，而非直接筛选小权重，因此小权重任务也已停止。中途结果全部不使用。只有weight1基准仍不理想时，才筛选0.1/0.25/0.5。

已排队的下一任务：FlexMatch权重筛选完成后，自动补跑早期FreeMatch soft-IF-w1、warmup1、lr=0.0046875的seed 4/5。脚本 `queue_freematch_warmup1_complete5.sh`，日志 `logs/freematch_warmup1_complete5_queue_20260721.log`，完成后生成 `results/freematch_warmup1_5seed_summary.csv`（test）和 `results/freematch_warmup1_5seed_val.csv`（val），完成标记 `FREEMATCH_WARMUP1_5SEED_DONE`。seed 1-3保持原checkpoint不重跑。

启动前已完成共享val阈值校准和配对分析。FreeMatch采用val平均F1最大化阈值后，IF在test五项指标均高于Base，暂不继续搜索权重；FlexMatch仍有Spe下降，故只对FlexMatch筛选更小IF权重。分析见 `results/freematch_paired_threshold_analysis.csv`、`results/flexmatch_paired_threshold_analysis.csv`。

## 已完成的修正版 ReFixMatch Base

任务：修正后的 ReFixMatch Base，BUS，5 seeds，最终只汇总 latest checkpoint。

- 启动脚本：`rerun_refixmatch_fixed_base.sh`
- 主日志：`logs/refixmatch_fixed_base_20260721_121817.log`
- 模型目录：`saved_models/usb_cv/refixmatch_fixed_base_s{1..5}`
- 预期汇总：`results/refixmatch_fixed_base_summary.csv`
- 完成标记：`REFIXMATCH_FIXED_BASE_DONE`
- GPU：RTX 3090；conda 环境：`wssl`。

截至 2026-07-21 15:21：

- seed 1：完成。
- seed 2：完成。
- seed 3：完成。
- seed 4：完成。
- seed 5：原会话在 2310 iteration 中断，已从 checkpoint 恢复并完成到 5500 iterations。
- 最终5-seed latest汇总已生成，`REFIXMATCH_FIXED_BASE_DONE` 已存在。

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

### 统一选择与报告规则（已与用户确认，不能事后更改）

- 判断规则用于选择“实验配置”，不是选择 checkpoint 或 iteration。
- 所有候选固定训练 5500 iterations；正式结果统一取每个 seed 的 `latest_model.pth`。
- 不用 `model_best.pth` 作为正式结果，不删除表现差的 seed，不混用 best/latest。
- 配置只在独立 `val` 上选择：验证 AUC 为首要指标，ACC/F1 为次要约束，同时检查 Sen/Spe 权衡。
- 先用 3 seeds 筛选；验证成立才补 seed 4/5。锁定配置后才在 `test` 上做一次最终评估。
- Base 与 +IF 必须使用相同 seed、训练长度、框架超参数和 checkpoint 口径。
- 当前一张 RTX 3090，任务串行执行，不并发抢显存。

### 优先级 1：完成 FreeMatch EMA 筛选

1. 完成当前 `ema_m=0.999` 的 Base vs soft-IF-w1 三种子配对实验：
   - 脚本：`run_freematch_ema999_val_screen.sh`；
   - Base：`fm_ema999_base_s{1..3}`；
   - IF：`fm_ema999_soft_w1_s{1..3}`；
   - IF 配置：soft target、weight=1、strong、`corrT=0.9`、zero5 warmup；
   - 只生成 `results/freematch_ema999_val_screen.csv`（`eval_dest=eval`）。
2. 与已有 `ema_m=0.0` 验证结果 `results/freematch_val_selection.csv` 比较。
3. 若 `ema_m=0.999` 的配对验证更好，补 seed 4/5；否则采用已有 `ema_m=0.0` soft-IF-w1 五种子，不再扩展。
4. 现有独立验证结果支持 soft-IF：Base/hard-IF/soft-IF AUC 分别为 0.861972/0.869038/0.870704，soft-IF 的验证 ACC/F1 也最高。

完成结论（2026-07-21 18:57）：EMA=0.999 的3-seed soft-IF val AUC为
0.872884±0.001804，与相同前3 seeds 的 EMA=0 soft-IF 0.872862±0.001552 实质持平；EMA=0 的ACC/F1略高且已有完整5 seeds。因此锁定 FreeMatch `ema_m=0` + soft-IF-w1 + zero5，不补 EMA=0.999 seed 4/5。

### 优先级 2：用已有 checkpoint 锁定 FlexMatch 配置

1. 在独立 `val` 上重新汇总以下五种子 `latest`，原则上无需重训：
   - `p_cutoff=0.9`：Base vs hard-IF；
   - `p_cutoff=0.95`：Base vs hard-IF vs soft-IF。
2. 以同阈值 Base->IF 的验证 AUC 增量为主，ACC/F1 和 Sen/Spe 权衡为辅，锁定阈值及 IF target。
3. 当前 test 结果只能用于盘点，不能用于选择；虽然 p=0.95 soft-IF 的 test AUC 较高，也必须由 val 结果确认。
4. 若优胜配置已有完整五种子 checkpoint，直接进入最终汇总，不重复训练。

完成结论（2026-07-21 19:38，均为5-seed latest val）：

- p=0.9 Base AUC 0.858711±0.004593；hard-IF 0.866472±0.003221，配对均值增量 +0.007761。
- p=0.95 Base AUC 0.863133±0.003708；hard-IF 0.868286±0.004031（+0.005153）；soft-IF 0.868796±0.004456（+0.005663）。
- 按预先锁定的“同阈值 Base->IF 验证AUC增量优先”规则，正式配置选 p=0.9 hard-IF；对应五种子checkpoint已齐，不重训。
- p=0.95 soft-IF 的绝对val AUC最高，但其Base->IF增量小于p=0.9 hard-IF；保留作消融，不因test结果改规则。
- 验证汇总：`results/flexmatch_val_selection.csv`。

### 优先级 3：修正版 ReFixMatch + IF

1. 修正版 p=0.9 Base 五种子已完成，latest 结果：AUC 0.838080±0.005114、ACC 0.776639±0.002766、Sen 0.720000±0.008724、Spe 0.835327±0.007946、F1 0.766375±0.003765。
2. 先在 Base 上选择 ReFixMatch 阈值，避免为 IF 调阈值：
   - 汇总现有修正版 p=0.9 Base 的三种子 val；
   - 新跑修正版 p=0.95 Base 三种子；
   - 只根据 Base 的 val 结果在 0.9/0.95 间锁定阈值。
3. 在锁定阈值上运行修正版 ReFixMatch+IFCF hard 三种子：weight=1、strong、`T=0.5`、`corrT=0.9`、zero5 warmup。
4. 若 val 上 Base->IF 有效，补 seed 4/5；若 hard IF 不理想，再把 soft IF 作为消融，不应一开始同时改多个因素。
5. 旧 `refixmatch_ifcf_hard_s*`/`soft_s*` 均来自错误实现，永不进入正式比较。

### 最终收尾

1. 为每个框架生成锁定配置下的五种子 latest 主表：AUC、ACC、Sen、Spe、F1 均值±标准差。
2. 计算同 seed Base-vs-IF 配对差值和显著性；不能声称所有指标普遍提高，需如实报告 Sen/Spe 权衡。
3. 最终五折交叉验证：每折固定训练长度，测试折不参与 checkpoint 或超参数选择。

## 安全与复现提醒

- 不删除或覆盖用户已有未提交修改。
- 不把 `model_best.pth` 用作正式主表结果。
- 不因看到测试指标而改变参数；参数必须由训练/验证数据决定。
- ReFixMatch 旧结果与修正版结果必须使用不同实验名。
- 普通 Codex 沙箱看不到 `/dev/nvidia*`；GPU 训练需获准在沙箱外运行，但仍使用同一个 `wssl` conda 环境。
