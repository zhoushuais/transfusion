# TransFusion KITTI Stage 1 定位诊断设计

## 1. 背景与目标

修复 KITTI 非方形 BEV 上 dense heatmap 的 `(x, y)` 目标顺序后，新的
Stage 1 仍完成了 40 epoch 但没有形成有效精度：Pedestrian/Cyclist 的严格
`AP_R40 3D` 全程为 0，Car 也接近 0。训练日志只能提供聚合损失和
`matched_ious`，无法区分故障首先发生在 dense heatmap、query 初始化、decoder
框回归还是最终分类分数。

本设计新增一个独立、只读的诊断工具，使用现有 Stage 1 checkpoint 在固定的
KITTI validation 子集上采集中间统计。目标是用一次短推理把故障范围缩小到明确
环节，为下一次代码修复提供证据。

本工具不训练模型，不修改 checkpoint，不改变模型 forward、loss、配置或评估
行为，也不启动 Stage 2。

## 2. 方案选择

采用独立脚本方案：

```text
projects/TransFusionKITTI/tools/diagnose_stage1.py
```

没有采用以下方案：

- 在模型内部临时增加 `print` 或 hook：会污染正式模型代码，输出也难以聚合。
- 直接进行学习率或损失消融：当前尚未确定失效环节，重新训练成本高且结论不清。

独立脚本复用生产模型的 bbox coder、local-max query 选择和 Hungarian assigner，
但不向模型模块写入状态。

## 3. 命令接口

脚本接受以下参数：

```text
config                 Stage 1 配置文件
--checkpoint           必填，待诊断的 Stage 1 checkpoint
--device               默认 cuda:0
--max-samples          默认 64
--output               必填，JSON 结果路径
```

服务器执行形式：

```bash
CUDA_VISIBLE_DEVICES=2 python \
  projects/TransFusionKITTI/tools/diagnose_stage1.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --checkpoint "work_dirs/transfusion_l_kitti_formal_run1_xyfix/best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_epoch_5.pth" \
  --max-samples 64 \
  --output work_dirs/transfusion_l_kitti_formal_run1_xyfix/stage1_diagnostics.json
```

`CUDA_VISIBLE_DEVICES=2` 后，进程内部仍使用默认的 `cuda:0`。

## 4. 数据流

### 4.1 诊断数据

以 `val_dataloader.dataset` 为基础构建顺序、无随机增强、`batch_size=1` 的诊断
dataloader。诊断 pipeline 显式加载点云和 3D 标注，并应用与 Stage 1 相同的
点云范围和目标范围过滤：

```text
LoadPointsFromFile
LoadAnnotations3D
PointsRangeFilter
ObjectRangeFilter
Pack3DDetInputs
```

使用 validation 的前 `max_samples` 个非空样本，固定顺序保证重复运行可比。不会
使用训练集的 ObjectSample、ObjectNoise、翻转或旋转增强。

### 4.2 模型执行

1. 按配置注册项目模块并构建 Stage 1 模型。
2. 严格检查 checkpoint 路径并通过 MMEngine 加载，模型设为 `eval()`。
3. 使用模型自身 data preprocessor 完成 voxelization。
4. 在 `torch.no_grad()` 下调用 tensor/raw forward，取得 bbox head 原始输出。
5. 只从张量副本计算统计，不调用 optimizer，不执行 backward。

脚本只接受 `fuse_img=False` 的 Stage 1 配置。若传入 Stage 2 配置则立即报错，
避免把 LiDAR-only 诊断结果与融合模型混用。

## 5. 诊断指标

所有指标同时给出 Overall 和 Pedestrian/Cyclist/Car 分类统计。距离统计至少输出
样本数、均值、中位数和 P90，召回率的分母是有效范围内的 GT 数量。

### 5.1 Dense heatmap 与初始 query

- GT 中心处的同类 dense heatmap sigmoid 分数。
- 生产逻辑 local-max 后，全类别 top-200 query 的类别数量分布。
- 每个 GT 到最近同类初始 query 的 BEV 网格距离。
- 同类 query 的 `recall@1/2/4/8 cells`。
- 没有任何同类 query 的 GT 数量。

这一组用于判断 dense heatmap 是否已经学到 GT 中心，以及 top-k/query 类别竞争
是否在 decoder 前就丢失目标。

### 5.2 Decoder 与 Hungarian 匹配

使用生产 `bbox_coder.decode()` 和生产 Hungarian assigner，对 decoder 输出与 GT
执行同口径匹配。对匹配对统计：

- BEV 中心距离，以及 `|dx|`、`|dy|`、`|dz|`。
- 三个尺寸分量的绝对误差。
- wrap 到 `[-pi, pi]` 后的 yaw 绝对误差。
- BEV IoU 和 3D IoU 的均值、中位数与 P90。
- 达到 KITTI 类别严格 IoU 阈值的比例：Car 为 0.7，Pedestrian/Cyclist 为 0.5。

这一组用于区分中心正确但高度/尺寸/朝向错误，与中心本身错误两种情况。

### 5.3 分类与最终分数

- Hungarian 匹配 query 的固定 `query_label` 与 GT 类别一致率。
- decoder 分类 sigmoid 分数统计。
- 与正式推理一致的最终分数：decoder sigmoid、dense query heatmap 分数和
  query one-hot mask 的乘积。
- 每类最终预测数量与分数分位数。

这一组用于判断框已经接近 GT，但被 query 类别锁定或最终分数乘法压低的情况。

## 6. 输出格式

终端打印紧凑摘要，便于用户直接粘贴；完整结果写为 JSON。JSON 顶层包含：

```text
metadata
ground_truth
dense_queries
decoder_matches
scores
```

`metadata` 至少记录配置路径、checkpoint 路径、设备、实际处理样本数、类别顺序、
BEV 特征尺寸、voxel size、out size factor 和 point cloud range。所有输出值必须是
JSON 原生数值，不保留 CUDA tensor 或 NumPy 标量。

## 7. 错误处理

以下情况以非零退出码失败，不输出容易误解的半成品结论：

- 配置或 checkpoint 不存在。
- 配置不是 `fuse_img=False` 的 Stage 1。
- checkpoint 关键模型参数无法加载。
- 实际 feature map 尺寸与配置推导尺寸不一致。
- 原始预测或解码框出现 NaN/Inf。
- 处理后没有样本或没有有效 GT。
- JSON 输出目录不可创建或结果无法写入。

个别样本没有 GT 时跳过并计数，不将其作为脚本失败。

## 8. 测试与验收

新增聚焦测试覆盖：

- 非方形 heatmap 的 flatten index 能还原正确 `(x, y)` query 坐标。
- synthetic heatmap 的同类最近距离与 `recall@1/2/4/8` 结果正确。
- yaw wrap、分位数和空类别统计不会产生 NaN。
- synthetic 匹配框的中心、尺寸和角度误差统计正确。
- 输出结构可被标准 `json.dumps` 序列化。
- Stage 2 配置会被拒绝，缺失 checkpoint 会明确报错。

本地验收运行项目测试、`compileall` 和 `git diff --check`。由于本地 Windows 环境
没有 H800 CUDA 扩展，真实 checkpoint 的数值诊断标记为服务器待验证；服务器只
需执行一次上述命令，不需要重跑 Stage 0 或 Stage 1。

## 9. 结果解释边界

诊断工具只定位当前 Stage 1 失效环节，不直接证明修复方案，也不产生论文实验
结果。后续只有在诊断证据指向明确代码错误、完成修复并重新训练后，才能判断
Stage 1 是否恢复。`xyfix` 继续保留为已验证的坐标一致性修复，但不表述为已经
恢复精度。
