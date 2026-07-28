# TransFusion KITTI Stage 1 梯度探针设计

## 1. 背景与当前证据

Stage 1 已完成 40 epoch FP32 正式训练，但 KITTI validation 没有形成有效
精度。现有只读诊断在 64 个 validation 样本上确认，失效最早出现在 dense
heatmap 与初始 query：

- GT 中心 dense heatmap 分数均值仅为 `0.0319`；
- Pedestrian/Cyclist 在 2 个 BEV cell 内的同类 query recall 均为 `0%`；
- top-200 query 并非完全被 Car 占据，三类累计数量分别为 944、1284、
  10572；
- Pedestrian/Cyclist 匹配框的尺寸误差明显呈 Car 尺寸倾向；
- 训练日志中的 `loss_heatmap` 从首轮早期的 `370.21` 快速下降到约
  `3.3`，随后长期停留在 `3.16-3.3`。

这些结果已定位失效环节，但尚不能区分以下原因：

1. heatmap head 没有获得有效梯度；
2. heatmap head 没有进入 optimizer，或 optimizer step 没有更新参数；
3. 梯度和更新链路正常，但随机初始化后的极大背景损失使优化过程快速塌缩；
4. heatmap target 的正样本虽存在，但其类别或数值分布异常。

因此新增一个独立梯度探针，在不修改正式 checkpoint 的前提下比较随机初始化
状态和 epoch 5 checkpoint 状态。

## 2. 目标与非目标

### 2.1 目标

- 使用 Stage 1 正式训练 pipeline 和同一批数据比较两个模型状态；
- 验证 heatmap head 的 `requires_grad`、optimizer 参数归属、梯度和参数更新；
- 记录每类 heatmap 正中心数量及正样本/背景预测概率；
- 将结果写成可复核的 JSON，并在终端输出高信号摘要；
- 为下一步单变量修复实验提供证据。

### 2.2 非目标

- 不启动正式训练，不恢复训练进度；
- 不覆盖或生成模型 checkpoint；
- 不修改 Stage 1/Stage 2 模型行为、loss 权重或初始化；
- 不用单批次参数变化量推断完整 40 epoch 的收敛效果；
- 不在本次探针中直接验证 heatmap bias 初始化等候选修复。

## 3. 工具接口

新增独立工具：

```text
projects/TransFusionKITTI/tools/probe_stage1_gradients.py
```

命令行接口：

```text
config                 Stage 1 配置文件
--checkpoint           必填，待比较的 Stage 1 checkpoint
--device               默认 cuda:0
--seed                 默认 0
--output               必填，JSON 输出路径
```

H800 预期命令：

```bash
CUDA_VISIBLE_DEVICES=2 python \
  projects/TransFusionKITTI/tools/probe_stage1_gradients.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --checkpoint "work_dirs/transfusion_l_kitti_formal_run1_xyfix/best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_epoch_5.pth" \
  --output work_dirs/transfusion_l_kitti_formal_run1_xyfix/stage1_gradient_probe.json
```

## 4. 数据与状态对比

探针使用 Stage 1 的真实 train pipeline，包括正式标注过滤和数据增强。为降低
非目标变量：

- batch size 保持正式配置中的 6；
- dataloader worker 改为 0，避免 worker 随机状态造成两个模型输入不同；
- 固定 Python、NumPy 和 PyTorch 随机种子；
- 两个模型状态复用同一份 CPU raw batch；
- 全程 FP32，不启用 AMP；
- 探针结束后不保留模型对象或优化器状态。

依次执行两个独立状态：

### 4.1 Fresh

- 按配置构建并初始化模型；
- 按配置构建 AdamW optimizer；
- 不加载任何 Stage 1 checkpoint。

### 4.2 Checkpoint

- 重新构建并初始化模型；
- 严格加载 epoch 5 `state_dict`；
- 按配置构建 optimizer，并加载 checkpoint 中的 optimizer state；
- 若 checkpoint 缺少 optimizer state，立即报错，不静默退化为新 optimizer。

## 5. 单状态探针流程

每个状态只执行一次前向和反向：

1. 使用模型自身 data preprocessor 处理同一 raw batch；
2. 前向得到 dense heatmap、query 和 decoder 输出；
3. 调用生产 `get_targets` 生成 heatmap target；
4. 计算生产 `loss_by_feat`，不复制一套诊断 loss 公式；
5. 使用 `torch.autograd.grad` 单独读取 `loss_heatmap` 对 heatmap head 的梯度；
6. 对总 loss 执行一次 backward，记录实际总梯度；
7. 对 heatmap head 参数执行一次诊断 optimizer step，比较 step 前后的参数。

正式 Stage 1 使用 `accumulative_counts=8`。本探针的参数 step 只用于证明“参数
在 optimizer 中且能够被更新”，不会模拟 8 个不同 micro-batch 的完整累积更新，
也不把参数变化量与正式训练单步幅度直接比较。总 loss 在探针 step 前按正式累积
因子缩放，并沿用配置中的梯度裁剪。

## 6. 输出指标

JSON 顶层包含：

```text
metadata
batch
fresh
checkpoint
comparison
```

### 6.1 Metadata 与 batch

- config、checkpoint、device、seed；
- batch size、正式累积因子和学习率；
- 每类 GT 数量；
- 每类 heatmap `target == 1` 的正中心数量；
- heatmap target 中 NaN/Inf 数量。

### 6.2 每个模型状态

- `loss_heatmap`、总 loss 及各生产 loss；
- dense heatmap 在正中心和背景位置的概率统计；
- 最后一层 heatmap bias 数值；
- heatmap head 每个参数的 `requires_grad`；
- heatmap head 每个参数是否属于 optimizer；
- heatmap-only 梯度范数和总 loss 梯度范数；
- 最终三类输出通道各自的 weight/bias 梯度范数；
- 梯度中的 NaN/Inf 数量；
- optimizer step 前后参数绝对变化范数和相对变化量；
- step 使用的学习率及梯度裁剪前范数。

### 6.3 Comparison

- fresh 与 checkpoint 的正中心概率变化；
- fresh 与 checkpoint 的 heatmap-only 梯度范数比值；
- 两个状态是否均满足“有梯度且参数发生更新”；
- checkpoint 是否表现为低概率、低空间响应的塌缩状态。

工具只输出事实型布尔量和数值，不自动宣称某个候选修复已经成立。

## 7. 错误处理与安全边界

以下情况立即失败并给出明确错误：

- 配置不是 `fuse_img=False` 的三类 KITTI Stage 1；
- checkpoint 不存在、模型参数无法严格加载或缺少 optimizer state；
- train batch 没有目标 GT 或没有 heatmap 正中心；
- heatmap head 参数未进入 optimizer；
- loss、梯度或参数变化出现 NaN/Inf；
- optimizer step 后所有 heatmap head 参数变化均为零。

探针不调用 checkpoint 保存逻辑，不写入 work directory 中除指定 JSON 之外的
任何文件。即使探针成功，也只表示诊断链路执行成功，不表示 Stage 1 已修复。

## 8. 测试与验证

本地测试覆盖：

- CLI 必填参数和 Stage 1 配置约束；
- raw batch 复用和随机种子约束；
- 正中心按类别统计；
- optimizer 参数归属检测；
- 梯度范数、非有限值检测和参数 delta；
- fresh/checkpoint comparison 生成；
- JSON 可序列化及失败条件。

本地环境不具备 H800、spconv 和真实 KITTI 数据，生产前向、CUDA backward、
checkpoint optimizer state 加载和实际数值均保持“待 H800 验证”。H800 验证只需
执行一次探针命令，不需要重跑 Stage 0、Stage 1 或 Stage 2。

## 9. 结果判读

- heatmap-only 梯度为零或缺失：继续追踪 loss 到 heatmap head 的 autograd 路径；
- 参数不在 optimizer 或 step delta 为零：修复 optimizer/冻结策略；
- fresh 初始背景概率高、loss 和梯度异常大，而 checkpoint 概率塌到约 0.02-0.04：
  再设计 heatmap 初始化的单变量短程验证；
- 梯度、optimizer 和更新均正常，但正中心类别或位置异常：回到 target 与训练增强
  后坐标的一致性检查；
- 所有链路正常且 target 正确：再使用固定小样本 overfit 试验研究优化动力学，
  仍不直接重跑 40 epoch。
