# TransFusion 现代框架迁移与 KITTI 多模态适配设计

## 1. 背景与目标

毕业论文研究内容二拟研究面向自动驾驶的多模态 3D 小目标检测，并与研究内容一形成以下递进关系：

- 研究内容一在纯视觉条件下增强小目标的多尺度特征表达和深度预测。
- 研究内容二进一步引入 LiDAR 几何信息，缓解远距离、小尺寸目标的单目深度歧义。

本项目以原版 TransFusion 的相机-LiDAR 模型为基线。原版代码依赖 MMDetection3D 0.11、MMDetection 2.x 和 MMCV 1.x，无法直接运行在目标服务器的 PyTorch 2.1.2、CUDA 11.8 和 H800 环境中；其官方相机-LiDAR 配置只覆盖 nuScenes 和 Waymo，也没有 KITTI 配置。

本设计的目标是在 MMDetection3D 1.4 中重建原版 TransFusion 的算法行为，并适配 KITTI 单前视相机和单帧 LiDAR，使其成为一个可训练、可评估、可继续开展小目标改进的干净基线。

## 2. 范围边界

### 2.1 本阶段包含

- 将原版 TransFusion 的 heatmap query 初始化、LiDAR BEV query 解码和 query-level 图像融合迁移到现代接口。
- 支持 KITTI `Car`、`Pedestrian`、`Cyclist` 三类联合训练。
- 支持 LiDAR-only TransFusion-L 和 camera-LiDAR TransFusion-LC 两种配置。
- 按原论文采用图像预训练、LiDAR-only 预训练、融合训练三个阶段。
- 使用现有 KITTI train/val 划分和 `AP_R40 3D` 评估。
- 支持单张 H800 的训练和测试。

### 2.2 本阶段不包含

- 不使用 BEVFusion 的 image-to-BEV 变换或 BEV 特征级融合替代原版 TransFusion。
- 不在迁移阶段加入任何小目标增强模块、重加权策略或新的融合方法。
- 不修改研究内容一的 MonoDETR、P1、P2、P3 代码和配置。
- 暂不要求 KITTI test server 提交结果。
- 不以框架迁移或 KITTI 数据适配本身作为论文创新点。

## 3. 固定环境契约

服务器目标环境固定为：

```text
Python 3.8
PyTorch 2.1.2 + CUDA 11.8
MMCV 2.1.0
MMEngine 0.10.x
MMDetection 3.2.0
MMDetection3D 1.4.0
spconv 2.x（CUDA 11.8 构建）
NVIDIA H800 80GB，单卡训练
```

项目不得依赖原仓库内置的旧 spconv、voxel、IoU CUDA 扩展。体素化、稀疏卷积和 IoU 计算均复用现代 MMCV、MMDetection3D 和 spconv 提供的实现，避免维护 PyTorch 1.x 的 `THC` 接口。

## 4. 代码组织

所有迁移代码放在现代 MMDetection3D 仓库的独立项目目录：

```text
projects/TransFusionKITTI/
├── configs/
│   ├── transfusion_l_kitti.py
│   ├── transfusion_lc_kitti.py
│   └── r50_fpn_kitti_2d.py
├── transfusion_kitti/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transfusion_detector.py
│   │   ├── transfusion_head.py
│   │   └── transfusion_bbox_coder.py
│   ├── datasets/
│       ├── __init__.py
│       └── transforms.py
│   └── evaluation/
│       ├── __init__.py
│       └── transfusion_kitti_metric.py
├── tools/
│   ├── convert_kitti_2d_to_coco.py
│   └── merge_pretrained_weights.py
├── tests/
└── README.md
```

配置通过 `custom_imports` 注册项目组件，不修改 MMDetection3D 核心注册表。`fuse_img=False` 必须完整回到 LiDAR-only 行为；`fuse_img=True` 才构建和调用图像分支。研究内容二的开关位于独立项目配置中，不向 `configs/monodetr.yaml` 引入与 MonoDETR 无关的配置。

## 5. 模型架构

### 5.1 TransFusion-L

```text
KITTI points [x, y, z, intensity]
  -> Det3DDataPreprocessor voxelization
  -> HardSimpleVFE
  -> SparseEncoder
  -> SECOND
  -> SECONDFPN
  -> shared BEV convolution
  -> dense class heatmap
  -> top-K heatmap proposals and class encoding
  -> LiDAR transformer decoder
  -> query prediction head
  -> KITTI LiDAR boxes
```

LiDAR query 主路径优先复用 MMDetection3D 1.4 `projects/BEVFusion` 中已经现代化的 `TransFusionHead` 实现，但只复用 TransFusionHead 的 LiDAR 查询代码，不使用 `BEVFusion` detector、DepthLSS、ConvFuser 或其多模态数据流。迁移后的类放入本项目命名空间，并保留来源说明。

### 5.2 TransFusion-LC

```text
TransFusion-L LiDAR queries
  + image_2 -> ResNet-50 -> FPN image feature
  -> image-guided dense heatmap branch
  -> predicted query center and 3D box corners projected by lidar2img
  -> visible-query mask and spatial Gaussian attention mask
  -> query-to-image cross-attention
  -> concatenate image query feature and LiDAR query feature
  -> fusion prediction head
  -> KITTI LiDAR boxes
```

需要保持的原版行为：

- 使用 LiDAR dense heatmap 和 image-guided dense heatmap 的平均分数选择 object query。
- image-guided dense heatmap 通过原版的“垂直压缩图像特征与 LiDAR BEV 特征 cross-attention”生成，不使用 LSS 或几何 lifting。
- LiDAR transformer decoder 先生成初步 query 和 3D 框，再把 query 中心及 3D 框角点投影到图像。
- 用投影框尺度构造 Gaussian spatial attention mask，限制 query 的图像 cross-attention 范围。
- 图像 query 与 LiDAR query 拼接后由融合预测头输出最终结果。
- 投影无效、位于相机后方或落在图像外的 query 回退到对应 LiDAR预测，不接受无效图像特征。
- 融合训练的 query 分类和回归损失只统计有效投影 query；无效 query 的最终预测用于回退，但不进入融合 query 损失。

`query_labels`、可见性 mask 和 dense heatmap 等逐 batch 数据必须随 head 输出显式传递，不能像原版一样缓存在模块可变成员上。图像位置编码缓存必须以特征形状、device 和 dtype 为键，避免不同 batch shape 或设备之间复用错误张量。

### 5.3 KITTI 回归码

KITTI 不预测速度，回归码固定为 8 维：

```text
[center_x, center_y, center_z, log_dim_x, log_dim_y, log_dim_z,
 sin(yaw), cos(yaw)]
```

解码后输出标准 7 维 LiDAR box。bbox coder 必须支持 batched encode/decode、空目标和不同设备，不得在 decode 中原地修改调用方张量。

## 6. KITTI 数据流

### 6.1 固定数据口径

- 使用与研究内容一一致的 KITTI train/val 划分。
- 类别采用现代 `KittiDataset` 的内部顺序：`Pedestrian`、`Cyclist`、`Car`；论文表格可重排为 `Car`、`Pedestrian`、`Cyclist`。
- 点云为单帧 4 维数据，不使用多 sweeps。
- 点云范围初始采用 `[0, -40, -3, 70.4, 40, 1]`。
- 体素大小初始采用 `[0.05, 0.05, 0.1]`。
- 图像只使用 `image_2`，不伪造多相机输入。

### 6.2 LiDAR-only 训练流水线

TransFusion-L 复用标准 KITTI 点云增强，包括数据库采样、ObjectNoise、BEV 水平翻转、全局旋转缩放、范围过滤和点打乱。该阶段不读取图像。

### 6.3 融合训练流水线

TransFusion-LC 同时读取：

```text
points
img
gt_bboxes_3d
gt_labels_3d
lidar2img
cam2img
lidar2cam
ori_shape / img_shape / pad_shape
homography_matrix
```

第一版融合基线按原论文关闭会破坏图像-点云对应关系的数据库采样、ObjectNoise、随机 BEV 翻转和全局旋转缩放。图像只做确定性 resize、normalize 和 pad，投影过程使用校准矩阵及图像 homography 更新坐标。

图像建议缩放到接近 KITTI 原始比例的 `1280 x 384` 上限并 pad 到 32 的倍数；阶段 0 和阶段 2 必须使用一致的归一化、FPN 输出层和 stride。

## 7. 三阶段训练与权重衔接

### 7.1 阶段 0：KITTI 2D 图像预训练

- 将 KITTI `label_2` 转换为 COCO 格式，过滤 `DontCare`，保留三类及原始 2D 框。
- 训练或微调 ResNet-50 + FPN 二维检测器。
- 输出只用于初始化 `img_backbone` 和 `img_neck`，二维检测头不进入 3D 融合模型。

### 7.2 阶段 1：KITTI TransFusion-L

- 使用 KITTI 标准 Voxel/SECOND/SECONDFPN 参数训练三类 LiDAR-only 模型。
- 使用 heatmap proposal initialization、Hungarian assignment、分类损失、框回归损失和 dense heatmap loss。
- 不包含 velocity head，`code_weights` 为 8 维。
- best checkpoint 按三类 `AP_R40 3D Moderate` 均值选择，同时完整记录三类 Easy、Moderate、Hard。

### 7.3 阶段 2：KITTI TransFusion-LC

- 通过合并脚本加载阶段 1 的完整 LiDAR 权重和阶段 0 的 `img_backbone`/`img_neck` 权重。
- 合并脚本打印匹配、缺失、形状不符和未使用 key；骨干出现非预期缺失时直接失败。
- 按原论文冻结 LiDAR backbone、LiDAR query 主路径、图像 backbone 和 FPN，并保持其 BN/归一化层为 eval 状态。
- 训练 image-guided heatmap、image-to-BEV query initialization、query-image cross-attention 和融合预测头。
- 首轮复现采用原论文 6 epoch 融合训练策略；单卡学习率按有效 batch size 线性缩放，必要时使用梯度累积保持固定有效 batch size。

## 8. 推理与评估

- 预测输出使用 `InstanceData` 和 `Det3DDataSample.pred_instances_3d`。
- 支持 batch size 大于 1，不保留原版只允许 batch size 1 的断言。
- 默认保持 TransFusion 的 NMS-free 输出；配置保留可选 rotated NMS，仅用于诊断，不进入默认基线。
- 使用现代 `KittiMetric` 计算 validation `AP_R40 3D`。
- 项目评估器在保留 `KittiMetric` 原始结果的同时，额外输出三类 `AP_R40 3D Moderate` 均值，供 checkpoint hook 选择 best checkpoint。
- 主表报告三类 Easy、Moderate、Hard，论文重点分析 Pedestrian 和 Cyclist，Car 作为辅助约束。
- 保存 KITTI 格式预测结果，便于定性可视化和后续 test server 扩展，但本阶段不要求提交 test server。

## 9. 稳健性与错误处理

必须显式覆盖以下情况：

- 校准矩阵缺失或形状错误时，在数据加载阶段给出样本 id 和字段名并失败。
- 投影深度小于等于零的 query 视为无效，不通过 clamp 把相机后方目标伪装为有效投影。
- 零个或一个 query 落入图像时仍可完成前向，不沿用原版 `<= 1` 时整批跳过的隐式行为。
- 一个 batch 中所有 query 都不可见时，输出退化为 LiDAR-only 路径，融合 query 损失安全归零，其他有效损失保持有限值。
- 空 GT 样本、仅 `DontCare` 样本和无正匹配样本的损失必须有限。
- 图像 resize/pad 后使用 homography 更新投影，训练和推理不得采用不同坐标口径。
- 冻结组件在每次 `train()` 后仍保持 `requires_grad=False` 和规范化层 eval 状态。

必要的稳健性修正必须在代码注释和迁移记录中标为“兼容性/正确性修复”，不能写成论文创新。

## 10. 验证门槛

实现按以下顺序验收，前一阶段失败时不得开始长训练。

### 10.1 单元测试

- bbox coder 8 维 encode/decode 往返测试。
- KITTI `lidar2img` 投影及图像 homography 测试。
- 可见、相机后方、图像外 query mask 测试。
- 无效 query 回退到 LiDAR预测测试。
- 零 GT、一个可见 query、全不可见 query 和 batch size 2 测试。
- `fuse_img=False` 不构建图像分支且可独立前向。

### 10.2 配置和模型构建

- 三个配置可由 MMEngine 加载。
- TransFusion-L 和 TransFusion-LC 可在 CPU 完成模型构建。
- 所有自定义组件均通过 `custom_imports` 注册，无需修改框架核心。

### 10.3 KITTI 单样本集成测试

- TransFusion-L 单样本 `loss` 和 `predict`。
- TransFusion-LC 单样本 `loss` 和 `predict`。
- 所有损失有限，输出框数量、维度、类别范围和坐标范围合法。
- 单次 backward 后只有预期参数产生梯度。

### 10.4 H800 smoke test

- 单卡完成 L 和 LC 各一个训练 iteration。
- LC 完成一次验证推理。
- 记录 CUDA、PyTorch、MMCV、spconv 版本和显存峰值。

### 10.5 小规模训练验证

- 在固定小子集上过拟合，确认 loss 明显下降并能输出接近 GT 的框。
- 完成短训练并跑通完整 KITTI validation 评估链路。

### 10.6 正式基线

- 阶段 0、1、2 均保存配置、commit、日志、best epoch 和 checkpoint 来源。
- 至少进行一次完整 L 和 LC 训练，确认 LC 相比 L 的结果可解释后，才开始研究内容二的小目标改进。
- 正式论文结果仍按三次独立运行取平均值；单次跑通结果只能标记为“已运行”，不能标记为“已验证有效”。

## 11. 完成定义

只有同时满足以下条件，才认为“框架迁移和 KITTI 适配完成”：

1. 现代环境不依赖旧 MMCV/MMDetection3D CUDA 扩展。
2. TransFusion-L 和 TransFusion-LC 均能在 KITTI 上训练、推理和计算 `AP_R40 3D`。
3. LC 保留原版 query-level 图像融合，不包含 BEVFusion 图像 BEV 融合路径。
4. L/LC 配置可复现实验且具有清晰的权重来源和冻结策略。
5. 单元测试、单样本测试、H800 smoke test 和短训练验证全部通过。
6. 迁移兼容性改动与后续论文创新代码能够明确区分。
