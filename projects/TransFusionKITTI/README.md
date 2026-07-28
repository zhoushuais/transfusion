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
64 样本短训练及 checkpoint 保存也已运行。TransFusion-LC 已完成正确加载
LiDAR/image 预训练权重后的单样本前反向、FP32 64 个唯一样本短训练及
checkpoint 保存，并在 3769 个 KITTI validation 样本上完成完整 AP11/AP40
评估。因此，现代框架迁移和 KITTI 多模态数据流已经跑通；单 H800 正式训练协议已经固化，但正式全量训练精度仍待验证，不能把 smoke 结果写成“多模态方法已验证有效”。

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

## Quick start (non-formal)

> **注意：** 本节的 Stage 0/1、checkpoint merge、Stage 2 和 evaluation
> 仅用于迁移验证与排障。这里的目录和 seed 不满足论文正式实验口径，不能把
> 输出写入论文结果；正式训练和评估必须使用后面的
> `Single-H800 formal training protocol`。

### Stage 0: image pretraining (quick start, non-formal)

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

### Stage 1: TransFusion-L (quick start, non-formal)

先做配置构建和单样本检查，再运行非正式 LiDAR-only 示例训练：

```bash
python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --device cuda:0 --mode backward

python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti
```

该示例和后面的正式基线均使用 FP32，与原版 TransFusion 配置一致。不要追加
`--amp`；迁移后的全局 autocast 会使 bbox 解码和 Hungarian 匹配出现半精度
数值溢出。

### Checkpoint merge (quick start, non-formal)

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

### Stage 2: TransFusion-LC (quick start, non-formal)

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

### Evaluation (quick start, non-formal)

```bash
python tools/test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  work_dirs/transfusion_lc_kitti/best_checkpoint.pth \
  --show-dir work_dirs/transfusion_lc_kitti/visualization
```

`Kitti metric/pred_instances_3d/KITTI/Overall_3D_AP40_moderate` 只用于
Stage 1/2 的 best checkpoint 选择，不是论文主指标。论文核心指标是
`Pedestrian` 和 `Cyclist` 的 KITTI AP_R40 3D Easy、Moderate、Hard；`Car`
和 Overall 作为辅助约束与报告项。这个 quick-start evaluation 只验证迁移
链路，不进入论文结果；论文正式结果仍需三次独立运行取平均。

## Single-H800 formal training protocol

单 H800 正式训练统一使用 FP32 和 MMEngine `OptimWrapper` 的梯度累积，
首次启动不要追加 `--amp` 或 `--auto-scale-lr`。三阶段批量口径如下：

| 阶段 | micro-batch | `accumulative_counts` | nominal batch |
|---|---:|---:|---:|
| Stage 0 image | 4 | 4 | 16 |
| Stage 1 LiDAR | 6 | 8 | 48 |
| Stage 2 LC | 2 | 8 | 16 |

nominal batch 只表示一次 optimizer update 汇总的样本数；BatchNorm 统计仍按
micro-batch 计算。Stage 1 因 epoch 尾部不能整除累计窗口，约 0.2% 的样本
存在 optimizer update 边界差异，因此单 H800 梯度累积不能写成与八卡 DDP
完全等价。

### Formal run1 preflight and config dump

run1 固定使用物理 GPU 2 和 seed 0。任何正式产物写入前，先确认 tracked
worktree 干净；`data` 等 untracked 文件不参与该检查。三个训练 work dir、
Stage 2 初始化文件、SHA-256 清单、两个评估目录和 run1 配置 dump 目录都必须
不存在，防止静默复用旧实验。通过检查后再创建目录，展开带完整 run1 覆盖项
的最终配置，并人工确认批量、累计步数、学习率缩放、最大学习率和 epoch。
本节三个严格 block 都在子 shell 中执行，失败只结束当前 block，不会关闭
交互式 SSH shell：

```bash
(
set -euo pipefail

TRACKED_CHANGES=$(git status --short --untracked-files=no)
if [[ -n "$TRACKED_CHANGES" ]]; then
  printf 'ERROR: tracked worktree is not clean:\n%s\n' \
    "$TRACKED_CHANGES" >&2
  exit 1
fi

FORMAL_PATHS=(
  work_dirs/r50_fpn_kitti_2d_formal_run1
  work_dirs/transfusion_l_kitti_formal_run1
  checkpoints/transfusion_kitti_stage2_formal_run1_init.pth
  work_dirs/transfusion_lc_kitti_formal_run1
  work_dirs/transfusion_kitti_formal_config_dumps/run1
  work_dirs/transfusion_kitti_formal_run1_sha256.txt
  work_dirs/transfusion_l_kitti_formal_run1_eval
  work_dirs/transfusion_lc_kitti_formal_run1_eval
)
for path in "${FORMAL_PATHS[@]}"; do
  if [[ -e "$path" ]]; then
    printf 'ERROR: formal run1 path already exists: %s\n' "$path" >&2
    exit 1
  fi
done

mkdir -p \
  work_dirs/r50_fpn_kitti_2d_formal_run1 \
  work_dirs/transfusion_l_kitti_formal_run1 \
  work_dirs/transfusion_lc_kitti_formal_run1 \
  work_dirs/transfusion_kitti_formal_config_dumps/run1 \
  checkpoints

CONFIG_DUMP_DIR=work_dirs/transfusion_kitti_formal_config_dumps/run1
python tools/misc/print_config.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py \
  --options \
    work_dir=work_dirs/r50_fpn_kitti_2d_formal_run1 \
    load_from=checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    randomness.seed=0 randomness.deterministic=False \
  > "$CONFIG_DUMP_DIR/stage0.py"
python tools/misc/print_config.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --options \
    work_dir=work_dirs/transfusion_l_kitti_formal_run1 \
    randomness.seed=0 randomness.deterministic=False \
  > "$CONFIG_DUMP_DIR/stage1.py"
python tools/misc/print_config.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --options \
    work_dir=work_dirs/transfusion_lc_kitti_formal_run1 \
    load_from=checkpoints/transfusion_kitti_stage2_formal_run1_init.pth \
    randomness.seed=0 randomness.deterministic=False \
  > "$CONFIG_DUMP_DIR/stage2.py"
grep -En 'batch_size|accumulative_counts|auto_scale_lr|eta_max|max_epochs' \
  "$CONFIG_DUMP_DIR"/stage{0,1,2}.py
)
```

### Formal run1 first launch

preflight 后每个阶段在对应 work dir 记录实际训练源码 commit。下面三个首次
训练命令均为 FP32，且不使用自动学习率缩放或断点恢复。

Stage 0：

```bash
git rev-parse HEAD \
  > work_dirs/r50_fpn_kitti_2d_formal_run1/source_commit.txt
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py \
  --work-dir work_dirs/r50_fpn_kitti_2d_formal_run1 \
  --cfg-options \
    load_from=checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    randomness.seed=0 randomness.deterministic=False
```

Stage 1：

```bash
git rev-parse HEAD \
  > work_dirs/transfusion_l_kitti_formal_run1/source_commit.txt
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti_formal_run1 \
  --cfg-options randomness.seed=0 randomness.deterministic=False
```

Stage 0 和 Stage 1 完成后，不假设 best checkpoint 的固定文件名。每一类
`best*.pth` 必须恰好有一个候选；零个或多个都中止并打印候选，不能任取一个
继续。严格检查通过后，在同一个 shell 中合并、记录三个文件的 SHA-256，
并校验 spconv2 metadata 及两个来源文件的路径和实际摘要：

```bash
(
set -euo pipefail

mapfile -t IMAGE_CANDIDATES < <(
  find work_dirs/r50_fpn_kitti_2d_formal_run1 \
    -maxdepth 1 -type f -name 'best*.pth' -print | LC_ALL=C sort
)
if (( ${#IMAGE_CANDIDATES[@]} != 1 )); then
  printf 'ERROR: expected exactly one image best checkpoint, found %d\n' \
    "${#IMAGE_CANDIDATES[@]}" >&2
  if (( ${#IMAGE_CANDIDATES[@]} == 0 )); then
    printf '  <none>\n' >&2
  else
    printf '  %s\n' "${IMAGE_CANDIDATES[@]}" >&2
  fi
  exit 1
fi

mapfile -t LIDAR_CANDIDATES < <(
  find work_dirs/transfusion_l_kitti_formal_run1 \
    -maxdepth 1 -type f -name 'best*.pth' -print | LC_ALL=C sort
)
if (( ${#LIDAR_CANDIDATES[@]} != 1 )); then
  printf 'ERROR: expected exactly one LiDAR best checkpoint, found %d\n' \
    "${#LIDAR_CANDIDATES[@]}" >&2
  if (( ${#LIDAR_CANDIDATES[@]} == 0 )); then
    printf '  <none>\n' >&2
  else
    printf '  %s\n' "${LIDAR_CANDIDATES[@]}" >&2
  fi
  exit 1
fi

export IMAGE_BEST="${IMAGE_CANDIDATES[0]}"
export LIDAR_BEST="${LIDAR_CANDIDATES[0]}"
STAGE2_INIT=checkpoints/transfusion_kitti_stage2_formal_run1_init.pth

python projects/TransFusionKITTI/tools/merge_pretrained_weights.py \
  --lidar "$LIDAR_BEST" \
  --image "$IMAGE_BEST" \
  --output "$STAGE2_INIT"

sha256sum "$IMAGE_BEST" "$LIDAR_BEST" "$STAGE2_INIT" \
  > work_dirs/transfusion_kitti_formal_run1_sha256.txt

python - <<'PY'
import hashlib
import os
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


checkpoint = torch.load(
    'checkpoints/transfusion_kitti_stage2_formal_run1_init.pth',
    map_location='cpu')
metadata = getattr(checkpoint['state_dict'], '_metadata', {})
assert metadata.get('middle_encoder.conv_input.0') == {'version': 2}

sources = checkpoint.get('meta', {}).get('transfusion_kitti_sources', {})
print('transfusion_kitti_sources:', sources)
for role, env_name in (('lidar', 'LIDAR_BEST'), ('image', 'IMAGE_BEST')):
    expected_path = Path(os.environ[env_name]).resolve()
    actual_path = Path(sources[role]['path']).resolve()
    print(f'{role} path: expected={expected_path} actual={actual_path}')
    assert actual_path == expected_path

    expected_digest = sources[role]['sha256']
    actual_digest = sha256(expected_path)
    print(
        f'{role} sha256: expected={expected_digest} '
        f'actual={actual_digest}')
    assert actual_digest == expected_digest
PY
)
```

Stage 2：

```bash
git rev-parse HEAD \
  > work_dirs/transfusion_lc_kitti_formal_run1/source_commit.txt
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --work-dir work_dirs/transfusion_lc_kitti_formal_run1 \
  --cfg-options \
    load_from=checkpoints/transfusion_kitti_stage2_formal_run1_init.pth \
    randomness.seed=0 randomness.deterministic=False
```

### Formal run1 resume

恢复训练只能使用首次启动时的原配置、原 work dir 和相同 cfg options。
MMEngine 会在指定 work dir 中优先恢复 `latest` checkpoint。禁止把
`*_smoke/epoch_1.pth` 当作正式训练恢复点。

Stage 0 恢复：

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py \
  --work-dir work_dirs/r50_fpn_kitti_2d_formal_run1 \
  --resume \
  --cfg-options \
    load_from=checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    randomness.seed=0 randomness.deterministic=False
```

Stage 1 恢复：

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti_formal_run1 \
  --resume \
  --cfg-options randomness.seed=0 randomness.deterministic=False
```

Stage 2 恢复：

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --work-dir work_dirs/transfusion_lc_kitti_formal_run1 \
  --resume \
  --cfg-options \
    load_from=checkpoints/transfusion_kitti_stage2_formal_run1_init.pth \
    randomness.seed=0 randomness.deterministic=False
```

### Formal run2 and run3 gate

只有 run1 的 loss 曲线、AP40、checkpoint 完整性和日志均审查通过后，才能
启动 run2 与 run3。run2 使用 seed 1，目录和初始化文件分别为
`work_dirs/r50_fpn_kitti_2d_formal_run2`、
`work_dirs/transfusion_l_kitti_formal_run2`、
`checkpoints/transfusion_kitti_stage2_formal_run2_init.pth` 和
`work_dirs/transfusion_lc_kitti_formal_run2`；run3 使用 seed 2，对应路径中的
`formal_run2` 全部替换为 `formal_run3`。除 seed 和 run 编号外，FP32、GPU、
配置展开、source commit、动态 best 查找、合并校验、SHA-256 记录和评估口径
均与 run1 相同。配置展开必须分别写入
`work_dirs/transfusion_kitti_formal_config_dumps/run2/` 和
`work_dirs/transfusion_kitti_formal_config_dumps/run3/`，并使用 seed 1 和 2；
每次都要先执行对应 run 的严格 preflight，不能覆盖 run1 或其他 run 的目录。

## Formal run1 evaluation

Stage 1 和 Stage 2 正式训练完成后，动态查找各自的 best checkpoint 并在
物理 GPU 2 上完整评估：

```bash
(
set -euo pipefail

mapfile -t LIDAR_CANDIDATES < <(
  find work_dirs/transfusion_l_kitti_formal_run1 \
    -maxdepth 1 -type f -name 'best*.pth' -print | LC_ALL=C sort
)
if (( ${#LIDAR_CANDIDATES[@]} != 1 )); then
  printf 'ERROR: expected exactly one LiDAR best checkpoint, found %d\n' \
    "${#LIDAR_CANDIDATES[@]}" >&2
  if (( ${#LIDAR_CANDIDATES[@]} == 0 )); then
    printf '  <none>\n' >&2
  else
    printf '  %s\n' "${LIDAR_CANDIDATES[@]}" >&2
  fi
  exit 1
fi

mapfile -t LC_CANDIDATES < <(
  find work_dirs/transfusion_lc_kitti_formal_run1 \
    -maxdepth 1 -type f -name 'best*.pth' -print | LC_ALL=C sort
)
if (( ${#LC_CANDIDATES[@]} != 1 )); then
  printf 'ERROR: expected exactly one LC best checkpoint, found %d\n' \
    "${#LC_CANDIDATES[@]}" >&2
  if (( ${#LC_CANDIDATES[@]} == 0 )); then
    printf '  <none>\n' >&2
  else
    printf '  %s\n' "${LC_CANDIDATES[@]}" >&2
  fi
  exit 1
fi

LIDAR_BEST="${LIDAR_CANDIDATES[0]}"
LC_BEST="${LC_CANDIDATES[0]}"

mkdir -p work_dirs/transfusion_l_kitti_formal_run1_eval
CUDA_VISIBLE_DEVICES=2 python tools/test.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  "$LIDAR_BEST" \
  --work-dir work_dirs/transfusion_l_kitti_formal_run1_eval

mkdir -p work_dirs/transfusion_lc_kitti_formal_run1_eval
CUDA_VISIBLE_DEVICES=2 python tools/test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  "$LC_BEST" \
  --work-dir work_dirs/transfusion_lc_kitti_formal_run1_eval
)
```

两次评估都要完整记录三类别 KITTI AP_R40 3D 的 Easy、Moderate、Hard。
论文核心指标是 `Pedestrian` 和 `Cyclist` 的这些指标；`Car` 作为辅助约束，
`Kitti metric/pred_instances_3d/KITTI/Overall_3D_AP40_moderate` 作为 Stage 1/2
best checkpoint 选择指标和辅助报告项，不是论文主指标。Stage 2 只与同一
run1 的 Stage 1 比较；smoke AP 只验证链路，不进入论文结果。

## H800 gradient-accumulation smoke test

此前的 FP32 64 样本 smoke 已验证现代框架迁移、KITTI 数据流、checkpoint
加载与评估链路。本节不是重复该结论，而是在梯度累积配置改动后、正式 run1
启动前进行一次三阶段验收：确认三份正式配置在单张 H800 上能够按 4/8/8 个
micro iteration 完成一次 optimizer update 并保存 checkpoint。smoke AP 只用于
检查执行链路，不能写入论文结果或用于论证方法有效性。

### Accumulation smoke preflight

在 H800 服务器的仓库根目录执行以下严格 preflight。`29837e49267312d7d34dfbae75d88fb110cfd0f0`
是已审查的 config/test/runbook 最低基线；后续纯文档修复可以位于其上，因此
只要求该提交是当前 `HEAD` 的 ancestor。preflight 会把真正用于本次 H800
实测的当前 SHA 同时打印并记录到
`work_dirs/transfusion_kitti_accum_smoke_commit.txt`。已有 smoke 目录不得删除、
清空或复用；任何一个已存在时，本次验收立即停止并改用经过记录的新后缀目录。

```bash
(
set -euo pipefail

EXPECTED_BRANCH=codex/transfusion-kitti-port
REVIEWED_BASELINE=29837e49267312d7d34dfbae75d88fb110cfd0f0
ACTUAL_BRANCH=$(git branch --show-current)
if [[ "$ACTUAL_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  printf 'ERROR: expected branch %s, got %s\n' \
    "$EXPECTED_BRANCH" "$ACTUAL_BRANCH" >&2
  exit 1
fi
if ! git merge-base --is-ancestor "$REVIEWED_BASELINE" HEAD; then
  printf 'ERROR: reviewed baseline %s is not an ancestor of HEAD\n' \
    "$REVIEWED_BASELINE" >&2
  exit 1
fi

WORKTREE_CHANGES=$(git status --porcelain)
if [[ -n "$WORKTREE_CHANGES" ]]; then
  printf 'ERROR: worktree is not clean:\n%s\n' \
    "$WORKTREE_CHANGES" >&2
  exit 1
fi

SMOKE_DIRS=(
  work_dirs/r50_fpn_kitti_2d_accum_smoke
  work_dirs/transfusion_l_kitti_accum_smoke
  work_dirs/transfusion_lc_kitti_accum_smoke
)
for path in "${SMOKE_DIRS[@]}"; do
  if [[ -e "$path" ]]; then
    printf 'ERROR: smoke directory already exists; do not delete or reuse it: %s\n' \
      "$path" >&2
    exit 1
  fi
done

for checkpoint in \
  checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
  checkpoints/transfusion_kitti_stage2_init.pth; do
  if [[ ! -f "$checkpoint" ]]; then
    printf 'ERROR: required local smoke checkpoint is missing: %s\n' \
      "$checkpoint" >&2
    exit 1
  fi
done

mkdir -p work_dirs
CURRENT_SHA=$(git rev-parse HEAD)
printf '%s\n' "$CURRENT_SHA" | \
  tee work_dirs/transfusion_kitti_accum_smoke_commit.txt
)
```

### Tests before training

先运行完整项目测试和 Python 编译检查。H800 环境已经配置 PyTorch、MMCV、
MMEngine、MMDetection、MMDetection3D 和 spconv；相关模块不能以依赖缺失为由
skip。以下 block 遇到测试 fail、任何 skip 或编译错误都会失败；此时停止，
不得开始训练。

```bash
(
set -euo pipefail

PYTEST_OUTPUT=$(mktemp)
trap 'rm -f "$PYTEST_OUTPUT"' EXIT
if ! python -m pytest -q projects/TransFusionKITTI/tests 2>&1 | \
    tee "$PYTEST_OUTPUT"; then
  printf 'ERROR: project tests failed; do not start smoke training\n' >&2
  exit 1
fi
if grep -Eq '[0-9]+ skipped' "$PYTEST_OUTPUT"; then
  printf 'ERROR: H800 project tests reported skips; inspect dependencies first\n' >&2
  exit 1
fi
python -m compileall -q projects/TransFusionKITTI
)
```

### Stage 0: image accumulation smoke

使用 GPU 2、seed 0 和本地 COCO checkpoint，取 16 个样本训练 1 epoch。不要
覆盖配置中的 `batch_size=4`：预期共 4 个 micro iteration，以
`accumulative_counts=4` 完成 1 次 optimizer update，并保存 `epoch_1.pth`。

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py \
  --work-dir work_dirs/r50_fpn_kitti_2d_accum_smoke \
  --cfg-options \
    load_from=checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    randomness.seed=0 randomness.deterministic=False \
    train_cfg.max_epochs=1 \
    val_cfg=None val_dataloader=None val_evaluator=None \
    train_dataloader.dataset.indices=16
```

### Stage 1: LiDAR accumulation smoke

仍使用 GPU 2 和 seed 0，取 24 个唯一训练样本。不要覆盖配置中的
`batch_size=6`；`RepeatDataset(times=2)` 后是 48 个样本，预期共 8 个 micro
iteration，以 `accumulative_counts=8` 完成 1 次 optimizer update，并保存
`epoch_1.pth`。

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti_accum_smoke \
  --cfg-options \
    randomness.seed=0 randomness.deterministic=False \
    train_cfg.max_epochs=1 \
    val_cfg=None val_dataloader=None val_evaluator=None \
    train_dataloader.dataset.dataset.indices=24
```

### Stage 2: LC accumulation smoke

使用 GPU 2 和 seed 0，加载此前已经验证可用的
`checkpoints/transfusion_kitti_stage2_init.pth`。该文件只作为 smoke 初始化，
不是 formal checkpoint，也不能代替 run1 的正式 Stage 0/1 合并产物。取 8 个
唯一训练样本且不覆盖配置中的 `batch_size=2`；repeat 后是 16 个样本，预期
共 8 个 micro iteration，以 `accumulative_counts=8` 完成 1 次 optimizer
update，并保存 `epoch_1.pth`。

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --work-dir work_dirs/transfusion_lc_kitti_accum_smoke \
  --cfg-options \
    load_from=checkpoints/transfusion_kitti_stage2_init.pth \
    randomness.seed=0 randomness.deterministic=False \
    train_cfg.max_epochs=1 \
    val_cfg=None val_dataloader=None val_evaluator=None \
    train_dataloader.dataset.dataset.indices=8
```

以上三个命令全部使用 FP32，不得追加 `--amp`、`--auto-scale-lr` 或
`--resume`。

### Accumulation smoke acceptance

MMEngine 0.10.7 会在每个 work dir 根目录保存以原配置 basename 命名的展开
配置。以下严格 block 检查三个 checkpoint、三份实际运行配置和所有日志；
Python 部分只使用标准库解析配置快照，不依赖 `rg` 或训练框架导入。

```bash
(
set -euo pipefail

SMOKE_DIRS=(
  work_dirs/r50_fpn_kitti_2d_accum_smoke
  work_dirs/transfusion_l_kitti_accum_smoke
  work_dirs/transfusion_lc_kitti_accum_smoke
)
for path in "${SMOKE_DIRS[@]}"; do
  if [[ ! -f "$path/epoch_1.pth" ]]; then
    printf 'ERROR: smoke checkpoint is missing: %s/epoch_1.pth\n' \
      "$path" >&2
    exit 1
  fi
done

python - <<'PY'
import ast
from pathlib import Path


def assignment(tree, name, path):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return node.value
    raise AssertionError(f'{path}: missing top-level assignment {name}')


def mapping_item(node, key, path, mapping_name):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'dict'):
        for keyword in node.keywords:
            if keyword.arg == key:
                return ast.literal_eval(keyword.value)
    elif isinstance(node, ast.Dict):
        for key_node, value_node in zip(node.keys, node.values):
            if ast.literal_eval(key_node) == key:
                return ast.literal_eval(value_node)
    raise AssertionError(f'{path}: missing {mapping_name}.{key}')


checks = (
    (Path('work_dirs/r50_fpn_kitti_2d_accum_smoke/'
          'r50_fpn_kitti_2d.py'), 4),
    (Path('work_dirs/transfusion_l_kitti_accum_smoke/'
          'transfusion_l_kitti.py'), 8),
    (Path('work_dirs/transfusion_lc_kitti_accum_smoke/'
          'transfusion_lc_kitti.py'), 8),
)
for path, expected in checks:
    if not path.is_file():
        raise AssertionError(f'missing MMEngine config snapshot: {path}')
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    actual = mapping_item(
        assignment(tree, 'optim_wrapper', path),
        'accumulative_counts', path, 'optim_wrapper')
    assert actual == expected, (
        f'{path}: accumulative_counts={actual}, expected {expected}')
    if expected == 8 and path.name == 'transfusion_lc_kitti.py':
        enabled = mapping_item(
            assignment(tree, 'auto_scale_lr', path),
            'enable', path, 'auto_scale_lr')
        assert enabled is False, f'{path}: auto_scale_lr.enable={enabled}'
    print(f'PASS: {path} accumulative_counts={actual}')
PY

mapfile -d '' STAGE0_LOGS < <(
  find work_dirs/r50_fpn_kitti_2d_accum_smoke \
    -type f -name '*.log' -print0
)
if (( ${#STAGE0_LOGS[@]} == 0 )); then
  printf 'ERROR: no Stage 0 smoke .log files found\n' >&2
  exit 1
fi
mapfile -d '' STAGE1_LOGS < <(
  find work_dirs/transfusion_l_kitti_accum_smoke \
    -type f -name '*.log' -print0
)
if (( ${#STAGE1_LOGS[@]} == 0 )); then
  printf 'ERROR: no Stage 1 smoke .log files found\n' >&2
  exit 1
fi
mapfile -d '' STAGE2_LOGS < <(
  find work_dirs/transfusion_lc_kitti_accum_smoke \
    -type f -name '*.log' -print0
)
if (( ${#STAGE2_LOGS[@]} == 0 )); then
  printf 'ERROR: no Stage 2 smoke .log files found\n' >&2
  exit 1
fi
ALL_LOGS=(
  "${STAGE0_LOGS[@]}"
  "${STAGE1_LOGS[@]}"
  "${STAGE2_LOGS[@]}"
)

require_log_pattern() {
  local description=$1
  local pattern=$2
  shift 2
  local status
  set +e
  grep -EinE -- "$pattern" "$@"
  status=$?
  set -e
  if (( status == 1 )); then
    printf 'ERROR: missing %s in smoke logs\n' "$description" >&2
    exit 1
  fi
  if (( status != 0 )); then
    printf 'ERROR: grep failed while checking %s (status=%s)\n' \
      "$description" "$status" >&2
    exit 1
  fi
}

reject_log_pattern() {
  local description=$1
  local pattern=$2
  shift 2
  local status
  set +e
  grep -EinE -- "$pattern" "$@"
  status=$?
  set -e
  if (( status == 0 )); then
    printf 'ERROR: detected %s in smoke logs\n' "$description" >&2
    exit 1
  fi
  if (( status != 1 )); then
    printf 'ERROR: grep failed while checking %s (status=%s)\n' \
      "$description" "$status" >&2
    exit 1
  fi
}

NUMERIC='[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?'
require_log_pattern 'Stage 0 numeric training loss' \
  "Epoch\\(train\\).*loss[^:=,[:space:]]*[[:space:]]*:[[:space:]]*$NUMERIC([^[:alnum:]_.+-]|$)" \
  "${STAGE0_LOGS[@]}"
require_log_pattern 'Stage 1 numeric training loss' \
  "Epoch\\(train\\).*loss[^:=,[:space:]]*[[:space:]]*:[[:space:]]*$NUMERIC([^[:alnum:]_.+-]|$)" \
  "${STAGE1_LOGS[@]}"
require_log_pattern 'Stage 2 numeric training loss' \
  "Epoch\\(train\\).*loss[^:=,[:space:]]*[[:space:]]*:[[:space:]]*$NUMERIC([^[:alnum:]_.+-]|$)" \
  "${STAGE2_LOGS[@]}"
require_log_pattern 'Stage 0 final iteration 4/4' \
  'Epoch\(train\)[[:space:]]+\[1\]\[4/4\]' \
  "${STAGE0_LOGS[@]}"
require_log_pattern 'Stage 1 final iteration 8/8' \
  'Epoch\(train\)[[:space:]]+\[1\]\[8/8\]' \
  "${STAGE1_LOGS[@]}"
require_log_pattern 'Stage 2 final iteration 8/8' \
  'Epoch\(train\)[[:space:]]+\[1\]\[8/8\]' \
  "${STAGE2_LOGS[@]}"
require_log_pattern 'Stage 1 numeric grad_norm' \
  "Epoch\\(train\\).*grad_norm[[:space:]]*:[[:space:]]*$NUMERIC([^[:alnum:]_.+-]|$)" \
  "${STAGE1_LOGS[@]}"
require_log_pattern 'Stage 2 numeric grad_norm' \
  "Epoch\\(train\\).*grad_norm[[:space:]]*:[[:space:]]*$NUMERIC([^[:alnum:]_.+-]|$)" \
  "${STAGE2_LOGS[@]}"
reject_log_pattern 'non-finite loss/grad_norm' \
  '(loss[^:=,[:space:]]*|grad_norm)["]?[[:space:]]*[:=][[:space:]]*[-+]?(nan|inf(inity)?)([^[:alpha:]]|$)' \
  "${ALL_LOGS[@]}"
reject_log_pattern 'automatic LR scaling' \
  'LR is set based on batch size|Scaling the original LR|automatically scaling (the )?(original )?(LR|learning rate)' \
  "${STAGE2_LOGS[@]}"
reject_log_pattern 'middle_encoder/backbone size mismatch' \
  '(middle_encoder|backbone).*size mismatch|size mismatch.*(middle_encoder|backbone)' \
  "${ALL_LOGS[@]}"

printf 'PASS: all three accumulation smoke artifacts and logs accepted\n'
)
```

Stage 1/2 配置显式启用了 `clip_grad`，因此必须有数值 `grad_norm` 证据。
Stage 0 继承的 Faster R-CNN `OptimWrapper` 没有 `clip_grad`，日志通常不输出
`grad_norm`；Stage 0 以数值 loss、`4/4` 迭代记录、配置快照和
`epoch_1.pth` 联合验收，不得为了 smoke 临时添加 `clip_grad`。

只有 preflight、零 skip 测试、编译、三阶段训练和 acceptance block 全部通过，
才能把 H800 gradient-accumulation smoke 标记为“已验证”并启动 formal run1。
任一阶段 OOM 时先保留原目录和日志，不得通过删除失败证据后重跑来掩盖问题。
允许评估的唯一 micro-batch/accumulation 备选是 Stage 0 `2/8`、Stage 1
`3/16`、Stage 2 `1/16`；任何改变都必须记录实际配置和 commit，并在同组正式
实验的所有 seed 中保持一致，不能只对某一次运行临时改值。

### Stage 1 gradient probe

仅当 `stage1_diagnostics.json` 已显示 dense GT-center score 异常偏低时运行
本探针。它使用 Stage 1 正式训练 pipeline 和同一个训练 raw batch，对比随机
初始化模型与指定 Stage 1 checkpoint。两种状态都会在内存中执行一次 FP32
诊断 optimizer step，但不会保存或覆盖任何模型 checkpoint。

checkpoint 必须包含正式训练保存的 AdamW optimizer state。MMEngine 的
`best_*.pth` 只保存模型权重，不满足本探针要求；当前 run1 应使用同一轮的
普通训练 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=2 python \
  projects/TransFusionKITTI/tools/probe_stage1_gradients.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --checkpoint work_dirs/transfusion_l_kitti_formal_run1_xyfix/epoch_5.pth \
  --output work_dirs/transfusion_l_kitti_formal_run1_xyfix/stage1_gradient_probe.json
```

终端出现 `STAGE1_GRADIENT_PROBE_OK` 只表示探针完整执行且 JSON 已写出，
不表示 Stage 1 已修复。检查 JSON 前不要启动 Stage 2，也不要重跑 40 epoch
Stage 1；下一步修复方案必须依据 fresh/checkpoint 的正中心概率、heatmap-only
梯度、optimizer 归属和参数变化量共同确定。

若要验证 dense heatmap 输出 bias 是否是训练初期塌缩的诱因，可额外对 fresh
模型设置一个诊断 bias。该选项只影响随机初始化的 fresh 模型，checkpoint 模型
仍按原值加载，不会修改训练代码或保存权重：

```bash
CUDA_VISIBLE_DEVICES=2 python \
  projects/TransFusionKITTI/tools/probe_stage1_gradients.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --checkpoint work_dirs/transfusion_l_kitti_formal_run1_xyfix/epoch_5.pth \
  --override-heatmap-bias -2.19 \
  --output work_dirs/transfusion_l_kitti_formal_run1_xyfix/stage1_gradient_probe_bias219.json
```

该命令的结果只能与不带该选项的原始 probe 对照解释：若 fresh 的初始
`loss_heatmap`、GT-center probability 和 heatmap-only gradient 同时显著改善，
`-2.19` 才能作为下一步短程 overfit 试验的候选初始化；这还不是正式修复结论。

### Stage 1 dense heatmap bias smoke

只有当 `stage1_gradient_probe_bias219.json` 显示 `-2.19` 能明显缓解 fresh
初始状态后，才运行本 smoke。该实验只验证训练侧初始化是否能阻止 Stage 1
dense heatmap 快速塌缩，不作为正式结果，也不要使用 `--resume` 覆盖旧目录。

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti_stage1_bias219_smoke \
  --cfg-options \
    model.bbox_head.dense_heatmap_init_bias=-2.19 \
    train_cfg.max_epochs=2 \
    train_dataloader.dataset.dataset.indices=64
```

训练结束后，对该 smoke checkpoint 再跑一次 probe。这里继续传
`--override-heatmap-bias -2.19`，只是为了让 fresh 对照模型和 smoke 训练初始
条件一致；checkpoint 本身仍按 `epoch_2.pth` 加载：

```bash
CUDA_VISIBLE_DEVICES=2 python \
  projects/TransFusionKITTI/tools/probe_stage1_gradients.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --checkpoint work_dirs/transfusion_l_kitti_stage1_bias219_smoke/epoch_2.pth \
  --override-heatmap-bias -2.19 \
  --output work_dirs/transfusion_l_kitti_stage1_bias219_smoke/stage1_gradient_probe_after_bias_smoke.json
```

跑完后检查并保留：

```bash
ls -lh work_dirs/transfusion_l_kitti_stage1_bias219_smoke/epoch_2.pth \
  work_dirs/transfusion_l_kitti_stage1_bias219_smoke/stage1_gradient_probe_after_bias_smoke.json
```

把 smoke 训练日志和
`stage1_gradient_probe_after_bias_smoke.json` 发回后，再判断是否进入短程
overfit 或停止 TransFusion 路线。
