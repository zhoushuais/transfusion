# TransFusion KITTI 现代框架迁移

本项目在 MMDetection3D 1.4 中重建原版 TransFusion 的 LiDAR query 与
query-level 相机-LiDAR 融合，并适配 KITTI `Pedestrian`、`Cyclist`、`Car`
三类别。代码独立放在 `projects/TransFusionKITTI`，没有修改
MMDetection3D 的模型结构，也没有使用 BEVFusion 的 LSS/BEV 图像融合；
核心目录仅包含一处 NumPy 兼容修复，将已删除的 `np.long` 替换为
语义等价的 `np.int64`。

当前状态：迁移源码、配置和工具已实现；已在 H800 上验证 MMCV CUDA
体素化、TransFusion-L 模型构建，以及真实 KITTI 单样本的前向、有限损失
和反向传播。TransFusion-L 的 FP32 64 样本短训练、恢复训练、普通与最佳
checkpoint 保存及完整 KITTI AP40 评估均已运行；Stage 0 图像分支的
64 样本短训练及 checkpoint 保存也已运行。正式全量训练精度和
TransFusion-LC 融合阶段仍待服务器验证；完成这些验证前，不能把完整
多模态基线写成“已跑通”或“已验证有效”。

## Environment check

在服务器的本仓库根目录创建独立环境：

```bash
conda create -n transfusion_kitti python=3.8 -y
conda activate transfusion_kitti

pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install openmim
mim install mmengine==0.10.7
mim install mmcv==2.1.0
pip install mmdet==3.2.0
pip install spconv-cu118==2.3.6 scipy==1.10.1
pip install -v -e .
```

确认版本和 H800：

```bash
python - <<'PY'
import torch
import mmcv
import mmengine
import mmdet
import mmdet3d

print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
print('torch arch list:', torch.cuda.get_arch_list())
print('mmcv:', mmcv.__version__)
print('mmengine:', mmengine.__version__)
print('mmdet:', mmdet.__version__)
print('mmdet3d:', mmdet3d.__version__)
PY
```

现代 MMCV 的体素化入口是 `mmcv.ops`：

```bash
python - <<'PY'
import torch
from mmcv.ops import Voxelization

voxelizer = Voxelization(
    voxel_size=[0.1, 0.1, 0.2],
    point_cloud_range=[-10, -10, -3, 10, 10, 3],
    max_num_points=10,
    max_voxels=1000)
points = torch.rand(1000, 5, device='cuda')
points[:, :3] = points[:, :3] * points.new_tensor([20, 20, 6]) \
    + points.new_tensor([-10, -10, -3])
voxels, coors, num_points = voxelizer(points)
print('CUDA voxelization passed:', voxels.shape, coors.shape, num_points.shape)
PY
```

## KITTI data preparation

目录至少包含：

```text
data/kitti/
├── ImageSets/train.txt
├── ImageSets/val.txt
├── training/image_2
├── training/label_2
├── training/calib
└── training/velodyne
```

`ImageSets/train.txt` 和 `val.txt` 必须使用与 MonoDETR 实验相同的划分。
生成现代 MMDetection3D 信息文件与降采样点云：

```bash
python tools/create_data.py kitti \
  --root-path data/kitti \
  --out-dir data/kitti \
  --extra-tag kitti
```

转换阶段 0 所需的 COCO 2D 标注：

```bash
python projects/TransFusionKITTI/tools/convert_kitti_2d_to_coco.py \
  --root data/kitti --split data/kitti/ImageSets/train.txt \
  --output data/kitti/annotations/kitti_2d_train.json
python projects/TransFusionKITTI/tools/convert_kitti_2d_to_coco.py \
  --root data/kitti --split data/kitti/ImageSets/val.txt \
  --output data/kitti/annotations/kitti_2d_val.json
```

## Stage 0: image pretraining

训练 KITTI 三类别 Faster R-CNN R50-FPN。阶段 2 只取其
`backbone.*` 和 `neck.*` 权重：

内网服务器先把官方 COCO 预训练权重放到：

```text
checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```

```bash
python tools/train.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py \
  --work-dir work_dirs/r50_fpn_kitti_2d \
  --cfg-options \
    load_from=checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```

COCO 的 80 类 bbox head 与 KITTI 三类别 head 尺寸不匹配是预期警告；
这些层会重新初始化，backbone 和 FPN 权重仍会正常加载。

## Stage 1: TransFusion-L

先做配置构建和单样本检查，再开始正式 LiDAR-only 训练：

```bash
python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --device cuda:0 --mode backward

python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti
```

正式基线使用 FP32，与原版 TransFusion 配置一致。当前不要追加 `--amp`；
迁移后的全局 autocast 会使 bbox 解码和 Hungarian 匹配出现半精度数值溢出。

## Checkpoint merge

先查看 MMEngine 实际保存的 best 文件名，不要假设固定别名：

```bash
find work_dirs/r50_fpn_kitti_2d -maxdepth 1 -name 'best*.pth'
find work_dirs/transfusion_l_kitti -maxdepth 1 -name 'best*.pth'
```

把命令中的两个路径替换为上一步真实输出：

```bash
python projects/TransFusionKITTI/tools/merge_pretrained_weights.py \
  --lidar work_dirs/transfusion_l_kitti/best_lidar_checkpoint.pth \
  --image work_dirs/r50_fpn_kitti_2d/best_image_checkpoint.pth \
  --output checkpoints/transfusion_kitti_stage2_init.pth
```

合并脚本会输出映射、跳过和覆盖数量，并把两个来源文件的 SHA-256 写入
checkpoint metadata，同时保留 spconv2 加载卷积核所需的
`state_dict._metadata`。出现零 image key 或形状冲突时会中止；如果日志仍有
`middle_encoder` 尺寸不匹配，不得在 `freeze_lidar=True` 下继续训练。

## Stage 2: TransFusion-LC

阶段 2 冻结 LiDAR 主干、LiDAR query 路径、图像 ResNet 和 FPN，只训练
原版 image-guided heatmap、query-image cross-attention 与融合预测头：

```bash
python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --device cuda:0 --mode backward \
  --checkpoint checkpoints/transfusion_kitti_stage2_init.pth

python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --work-dir work_dirs/transfusion_lc_kitti \
  --cfg-options load_from=checkpoints/transfusion_kitti_stage2_init.pth
```

## Evaluation

```bash
python tools/test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  work_dirs/transfusion_lc_kitti/best_checkpoint.pth \
  --show-dir work_dirs/transfusion_lc_kitti/visualization
```

主指标是 `Kitti metric/pred_instances_3d/KITTI/Overall_3D_AP40_moderate`，
同时完整记录
三类别 3D AP40 的 Easy、Moderate、Hard。论文正式结果仍需三次独立运行取
平均，单次跑通只能记为“已运行”。

## H800 smoke test

依次执行：

```bash
python -m pytest -q projects/TransFusionKITTI/tests
python -m compileall -q projects/TransFusionKITTI

python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --device cuda:0 --mode backward
python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --device cuda:0 --mode predict \
  --checkpoint checkpoints/transfusion_kitti_stage2_init.pth
```

再分别用固定的前 64 个训练样本跑 1 epoch。MMEngine 会在最后一个 epoch
强制执行 validation，因此这里需要同时将三个验证配置设为 `None`：

```bash
python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti_smoke \
  --cfg-options train_cfg.max_epochs=1 \
    val_cfg=None val_dataloader=None val_evaluator=None \
    train_dataloader.batch_size=1 \
    train_dataloader.dataset.dataset.indices=64

python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --work-dir work_dirs/transfusion_lc_kitti_smoke \
  --cfg-options load_from=checkpoints/transfusion_kitti_stage2_init.pth \
    train_cfg.max_epochs=1 \
    val_cfg=None val_dataloader=None val_evaluator=None \
    train_dataloader.batch_size=1 \
    train_dataloader.dataset.dataset.indices=64
```

全部通过后，再运行一次完整 validation；在此之前不要启动三次正式长训练。
