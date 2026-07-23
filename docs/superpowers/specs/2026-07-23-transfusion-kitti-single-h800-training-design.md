# TransFusion KITTI 单 H800 正式训练设计

## 1. 背景与目标

TransFusion KITTI 的现代框架迁移、三阶段权重衔接、FP32 短训练和完整
KITTI validation 已在单张 H800 上跑通。下一步是运行可进入实验记录的正式
基线，而不是继续使用 64 个唯一样本的 smoke checkpoint。

现有三个阶段的学习率分别按原配置的全局 batch 设计：Stage 0 为 16，
Stage 1 为 48，Stage 2 为 16。目标服务器只有一张 H800。本设计采用梯度
累积维持名义等效全局 batch，并保留现有 optimizer 学习率和完整调度曲线。

本设计只确定训练协议，不改变模型结构、数据划分、损失、冻结策略或评估
指标，也不引入任何论文创新模块。

本设计覆盖 2026-07-22 迁移设计第 7.3 节中“单卡线性缩放学习率”的旧
口径；正式单 H800 实验统一采用本设计的梯度累积方案。

## 2. 已批准方案

| 阶段 | 配置 | 单卡 micro-batch | 累积次数 | 名义等效 batch | epochs |
|---|---|---:|---:|---:|---:|
| Stage 0 | `r50_fpn_kitti_2d.py` | 4 | 4 | 16 | 12 |
| Stage 1 | `transfusion_l_kitti.py` | 6 | 8 | 48 | 40 |
| Stage 2 | `transfusion_lc_kitti.py` | 2 | 8 | 16 | 6 |

使用 MMEngine `OptimWrapper.accumulative_counts` 完成梯度累积。三个阶段均
使用 FP32，不启用 `--amp`。

## 3. 配置行为

### 3.1 Stage 0

- 保持 Faster R-CNN 的单卡 `batch_size=4` 和 12 epoch schedule。
- 在 `optim_wrapper` 中设置 `accumulative_counts=4`。
- 保持 `auto_scale_lr.enable=False`。
- 从服务器本地 COCO Faster R-CNN checkpoint 初始化。
- best checkpoint 继续按 `coco/bbox_mAP` 选择。

### 3.2 Stage 1

- 保持 KITTI `RepeatDataset(times=2)`、单卡 `batch_size=6` 和 40 epoch
  cyclic schedule。
- 在 `optim_wrapper` 中设置 `accumulative_counts=8`。
- 保持 optimizer LR、两个 CosineAnnealingLR 阶段和 momentum schedule
  不变。
- 保持 `auto_scale_lr.enable=False`。
- best checkpoint 继续按三类 strict `AP_R40 3D Moderate` 均值选择。

### 3.3 Stage 2

- 保持单卡 `batch_size=2`、6 epoch OneCycle schedule 和原有冻结策略。
- 在 `optim_wrapper` 中设置 `accumulative_counts=8`。
- 将 `auto_scale_lr.enable` 设为 `False`。
- 保持 optimizer `lr=1e-4` 和 OneCycle `eta_max=1e-3` 不变。
- 从正式 Stage 0/1 best checkpoint 的合并结果初始化。
- best checkpoint 继续按三类 strict `AP_R40 3D Moderate` 均值选择。

关闭 Stage 2 自动缩放是必要的：MMEngine 0.10.7 的 `scale_lr` 只修改
optimizer parameter group 的当前 LR，不会同步缩放 OneCycle 的绝对
`eta_max`。在梯度累积已经恢复名义全局 batch 后，继续自动缩放既不必要，
也会造成日志口径与实际 OneCycle 曲线不一致。

## 4. 等效性边界

梯度累积使每次 optimizer update 汇总的名义样本数与原全局 batch 一致，
学习率仍按样本在 epoch 内的进度变化。它不是逐指令等同于八卡 DDP：

- Stage 0 和 Stage 2 的训练样本数可被等效 batch 整除，更新边界明确。
- Stage 1 的 `RepeatDataset` 长度为 7424，不能被 48 整除；单卡 dataloader
  与多卡 sampler 对 epoch 尾部样本的补齐方式不同，optimizer update 总数
  存在约 0.2% 的边界差异。
- BatchNorm 统计基于 micro-batch，而不是累积后的名义 batch；这与常规
  非 SyncBN 的多卡训练一致，但不能描述为“大 batch BN”。

论文和实验记录统一使用“梯度累积后的名义等效 batch”，不写成“与八卡
训练完全等价”。

## 5. 正式训练数据流

正式 run1 固定 `seed=0`，使用以下独立产物，禁止复用 smoke work dir：

```text
work_dirs/r50_fpn_kitti_2d_formal_run1/
work_dirs/transfusion_l_kitti_formal_run1/
checkpoints/transfusion_kitti_stage2_formal_run1_init.pth
work_dirs/transfusion_lc_kitti_formal_run1/
```

顺序固定为：

1. Stage 0 使用本地 COCO checkpoint 完成 12 epoch 训练并选出 best。
2. Stage 1 从头完成 40 epoch LiDAR-only 训练并选出 best。
3. 合并 Stage 0 和 Stage 1 best checkpoint，并验证 spconv2 metadata。
4. Stage 2 从合并 checkpoint 完成 6 epoch 融合训练并选出 best。
5. 使用 Stage 1 和 Stage 2 best checkpoint 分别运行完整 KITTI validation。

Stage 0 与 Stage 1 在算法上彼此独立，但单卡服务器按上述顺序串行执行，
避免资源竞争。三次正式运行固定使用 `seed=0/1/2`，分别对应 run1/run2/run3；
run2 和 run3 只在 run1 完成并通过结果审查后启动，并使用同样的目录命名规则。

## 6. Checkpoint 与恢复规则

- 每个正式阶段使用新的 work dir，初次启动不加 `--resume`。
- 训练意外中断时，只能在同一配置、同一 work dir 下使用 `--resume`。
- 不从 smoke `epoch_1.pth` 恢复正式训练。
- Stage 2 只接受正式 Stage 0/1 best checkpoint 的合并文件。
- 合并后必须确认 `middle_encoder.conv_input.0` metadata 为
  `{'version': 2}`，且加载日志没有 `middle_encoder` size mismatch。
- 每个阶段记录 commit、配置快照、seed、日志、best epoch、checkpoint 路径
  和来源 SHA-256。

## 7. 测试与验收

### 7.1 配置测试

测试加载三个配置并断言：

- `(batch_size, accumulative_counts)` 分别为 `(4, 4)`、`(6, 8)`、
  `(2, 8)`。
- 名义等效 batch 分别为 16、48、16。
- Stage 2 `auto_scale_lr.enable=False`。
- epochs、LR、OneCycle `eta_max` 和 best metric 未被意外改变。

### 7.2 累积 smoke test

在正式长训练前，用固定小子集各运行一次启用梯度累积的短训练。验收条件：

- 配置 dump 包含预期的 `accumulative_counts`。
- loss 和 grad norm 有限，无 NaN/Inf。
- optimizer 能更新可训练参数并保存 checkpoint。
- Stage 2 不再打印自动 LR 缩放提示。
- LC 加载时只允许预期的新融合模块 missing keys，不允许 backbone 或
  `middle_encoder` shape mismatch。

### 7.3 正式 run1

- 三阶段均完成预定 epochs，validation 和 best checkpoint 保存正常。
- Stage 0 bbox mAP、Stage 1/2 KITTI AP40 随训练形成可解释趋势。
- Stage 2 正式 AP 只与同一 run1 的 Stage 1 比较。
- smoke AP 不进入任何论文表格。

## 8. 失败处理

- 如果发生 OOM，先停止训练并保留日志，不直接降低等效 batch。
- 允许的备选 micro-batch/累积组合必须保持乘积不变：Stage 0 可用
  `(2, 8)`，Stage 1 可用 `(3, 16)`，Stage 2 可用 `(1, 16)`。
- micro-batch 改变会影响归一化统计，必须在实验记录中注明，且同一组可比
  实验必须使用相同组合。
- 如果正式训练出现非有限 loss，先复现并定位，不通过 AMP、跳过 batch 或
  降低匹配代价掩盖问题。

## 9. 完成定义

本设计完成需要同时满足：

1. 三个配置固化并测试通过。
2. 启用梯度累积的短训练在 H800 上通过。
3. 正式 run1 的 Stage 0、Stage 1、checkpoint merge 和 Stage 2 全部完成。
4. Stage 1/2 best checkpoint 的完整 KITTI AP40 已记录并可比较。
5. 所有正式产物与 smoke 产物隔离，训练口径可从配置和日志复现。
