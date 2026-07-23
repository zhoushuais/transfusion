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

再分别用固定的前 64 个唯一训练样本跑 1 epoch。基础 KITTI 配置使用
`RepeatDataset(times=2)`，因此日志会显示 128 iterations。MMEngine 会在
最后一个 epoch 强制执行 validation，因此这里需要同时将三个验证配置设为
`None`：

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

全部通过后，再运行一次完整 validation。冒烟 checkpoint 的指标只用于验证
评估链路，不进入论文结果；完整 validation 通过后才开始正式全量训练。
