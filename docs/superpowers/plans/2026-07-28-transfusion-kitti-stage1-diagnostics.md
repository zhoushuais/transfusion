# TransFusion KITTI Stage 1 Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个独立、只读的 Stage 1 诊断脚本，使用现有 KITTI LiDAR checkpoint 定位失效首先发生在 dense heatmap、初始 query、decoder 框匹配还是最终分类分数。

**Architecture:** 所有实现集中在 `diagnose_stage1.py`，但按纯统计函数、单样本诊断、运行时装配和结果序列化四个边界组织。纯函数使用合成张量测试；服务器运行时复用生产 `TransFusionKITTIHead` 的 local-max、bbox coder、Hungarian assigner 和正式 `predict_by_feat()`，且全程 `eval()`、`torch.no_grad()`，不写入模型状态。

**Tech Stack:** Python 3.8、PyTorch 2.1、MMEngine 0.10、MMCV 2.1、MMDetection 3.2、MMDetection3D 1.4、pytest、SciPy Hungarian assignment、MMCV rotated IoU CUDA op。

---

## File Structure

- Create: `projects/TransFusionKITTI/tools/diagnose_stage1.py`
  - CLI、诊断 dataloader、模型/checkpoint 装配、纯统计函数、逐样本诊断、聚合、终端摘要和 JSON 写入。
- Create: `projects/TransFusionKITTI/tests/test_diagnose_stage1.py`
  - 非方形网格坐标、query recall、分位数、角度、匹配误差、配置拒绝、CLI 和 JSON 序列化测试。

不修改 `transfusion_head.py`、`transfusion_detector.py`、任何 config 或训练/评估入口。诊断工具不是 P1/P2/P3 研究模块，因此不使用这些方向标记，避免把研究内容一的标签误写到研究内容二的工程诊断代码中。

### Task 1: CLI、输入契约与诊断 dataloader

**Files:**
- Create: `projects/TransFusionKITTI/tools/diagnose_stage1.py`
- Create: `projects/TransFusionKITTI/tests/test_diagnose_stage1.py`

- [ ] **Step 1: 写 CLI 与配置契约失败测试**

在测试文件中先加入以下测试。它们固定 checkpoint/output 必填、Stage 2 拒绝和验证集无增强 pipeline：

```python
from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('mmengine')
pytest.importorskip('mmcv')
pytest.importorskip('mmdet')

from mmengine.config import ConfigDict

from projects.TransFusionKITTI.tools import diagnose_stage1


def test_parser_requires_checkpoint_and_output():
    parser = diagnose_stage1.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['config.py'])
    args = parser.parse_args([
        'config.py', '--checkpoint', 'epoch_5.pth', '--output', 'diag.json'
    ])
    assert args.device == 'cuda:0'
    assert args.max_samples == 64


def test_validate_stage1_rejects_fusion_config():
    cfg = ConfigDict(model=ConfigDict(fuse_img=True))
    with pytest.raises(ValueError, match='fuse_img=False'):
        diagnose_stage1.validate_stage1_config(cfg)


def test_build_diagnostic_dataloader_cfg_is_ordered_and_unaugmented():
    cfg = ConfigDict(
        backend_args=None,
        point_cloud_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
        val_dataloader=ConfigDict(
            batch_size=1,
            num_workers=1,
            persistent_workers=True,
            sampler=ConfigDict(type='DefaultSampler', shuffle=False),
            dataset=ConfigDict(
                type='KittiDataset',
                test_mode=True,
                pipeline=[ConfigDict(type='Pack3DDetInputs')],
            ),
        ),
    )
    result = diagnose_stage1.build_diagnostic_dataloader_cfg(cfg)
    assert result.batch_size == 1
    assert result.num_workers == 0
    assert result.persistent_workers is False
    assert result.sampler.shuffle is False
    assert result.dataset.test_mode is False
    assert result.dataset.filter_empty_gt is False
    assert [item.type for item in result.dataset.pipeline] == [
        'LoadPointsFromFile',
        'LoadAnnotations3D',
        'PointsRangeFilter',
        'ObjectRangeFilter',
        'Pack3DDetInputs',
    ]
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run:

```bash
python -m pytest -q \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
```

Expected: collection fails with `cannot import name 'diagnose_stage1'` or the first missing API.

- [ ] **Step 3: 实现 CLI、Stage 1 校验和 dataloader 配置构造**

在脚本中加入以下实现。使用深拷贝，保证不会修改已加载的正式配置：

```python
"""Diagnose a LiDAR-only TransFusion KITTI Stage-1 checkpoint."""

import argparse
import copy
from pathlib import Path


CLASS_NAMES = ('Pedestrian', 'Cyclist', 'Car')
RECALL_RADII = (1.0, 2.0, 4.0, 8.0)
STRICT_IOU = {'Pedestrian': 0.5, 'Cyclist': 0.5, 'Car': 0.7}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def validate_stage1_config(cfg) -> None:
    if bool(cfg.model.get('fuse_img', False)):
        raise ValueError(
            'Stage 1 diagnostics require model.fuse_img=False')
    if int(cfg.model.bbox_head.get('num_classes', -1)) != len(CLASS_NAMES):
        raise ValueError('Stage 1 diagnostics require exactly 3 KITTI classes')


def build_diagnostic_dataloader_cfg(cfg):
    dataloader_cfg = copy.deepcopy(cfg.val_dataloader)
    dataloader_cfg.batch_size = 1
    dataloader_cfg.num_workers = 0
    dataloader_cfg.persistent_workers = False
    dataloader_cfg.sampler.shuffle = False
    dataset_cfg = dataloader_cfg.dataset
    dataset_cfg.test_mode = False
    dataset_cfg.filter_empty_gt = False
    backend_args = cfg.get('backend_args', None)
    point_cloud_range = list(cfg.point_cloud_range)
    dataset_cfg.pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            backend_args=backend_args,
        ),
        dict(type='LoadAnnotations3D', with_bbox_3d=True,
             with_label_3d=True),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(
            type='Pack3DDetInputs',
            keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        ),
    ]
    return dataloader_cfg
```

- [ ] **Step 4: 运行 Task 1 测试**

Run:

```bash
python -m pytest -q \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
```

Expected: `3 passed`。

- [ ] **Step 5: 提交 Task 1**

```bash
git add projects/TransFusionKITTI/tools/diagnose_stage1.py \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
git commit -m "test: define Stage 1 diagnostic contracts"
```

### Task 2: 统计原语、非方形坐标和 query 诊断

**Files:**
- Modify: `projects/TransFusionKITTI/tools/diagnose_stage1.py`
- Modify: `projects/TransFusionKITTI/tests/test_diagnose_stage1.py`

- [ ] **Step 1: 写统计与 query 选择失败测试**

追加：

```python
import math

import torch


def test_flat_indices_restore_xy_on_non_square_map():
    indices = torch.tensor([0, 4, 5, 13])
    result = diagnose_stage1.flat_indices_to_xy(indices, width=5)
    torch.testing.assert_close(
        result,
        torch.tensor([[0., 0.], [4., 0.], [0., 1.], [3., 2.]]),
    )


def test_select_queries_preserves_class_and_xy():
    localized = torch.zeros(1, 2, 3, 5)
    localized[0, 0, 1, 4] = 0.9
    localized[0, 1, 2, 1] = 0.8
    selected = diagnose_stage1.select_queries(localized, num_proposals=2)
    assert selected['labels'][0].tolist() == [0, 1]
    torch.testing.assert_close(
        selected['xy'][0], torch.tensor([[4., 1.], [1., 2.]]))


def test_query_metrics_use_same_class_distance():
    query_xy = torch.tensor([[4., 1.], [1., 2.], [0., 0.]])
    query_labels = torch.tensor([0, 1, 0])
    gt_xy = torch.tensor([[3., 1.], [4., 2.], [1., 2.]])
    gt_labels = torch.tensor([0, 0, 1])
    result = diagnose_stage1.query_metrics(
        query_xy, query_labels, gt_xy, gt_labels)
    torch.testing.assert_close(
        result['nearest_distance'], torch.tensor([1., 1., 0.]))
    assert result['recall'][1.0].tolist() == [True, True, True]
    assert result['no_same_class'].tolist() == [False, False, False]


def test_numeric_summary_empty_and_finite_values_are_json_safe():
    assert diagnose_stage1.numeric_summary([]) == {
        'count': 0, 'mean': None, 'median': None, 'p90': None
    }
    result = diagnose_stage1.numeric_summary([1.0, 2.0, 3.0])
    assert result['count'] == 3
    assert result['mean'] == pytest.approx(2.0)
    assert result['median'] == pytest.approx(2.0)
    assert math.isfinite(result['p90'])


def test_wrap_angle_returns_small_absolute_boundary_error():
    error = diagnose_stage1.absolute_yaw_error(
        torch.tensor([math.pi - 0.1]),
        torch.tensor([-math.pi + 0.1]),
    )
    torch.testing.assert_close(error, torch.tensor([0.2]), atol=1e-6,
                               rtol=1e-6)
```

- [ ] **Step 2: 运行新增测试并确认缺失函数失败**

Run:

```bash
python -m pytest -q \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py \
  -k "flat_indices or select_queries or query_metrics or numeric_summary or wrap_angle"
```

Expected: FAIL，指出 `flat_indices_to_xy` 等函数尚不存在。

- [ ] **Step 3: 实现统计与 query 纯函数**

追加以下实现：

```python
from typing import Dict, Iterable

import torch
from torch import Tensor


def _require_finite(name: str, value: Tensor) -> None:
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise RuntimeError(f'{name} contains NaN or Inf')


def numeric_summary(values: Iterable[float]) -> dict:
    tensor = torch.as_tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return dict(count=0, mean=None, median=None, p90=None)
    _require_finite('summary values', tensor)
    return dict(
        count=int(tensor.numel()),
        mean=float(tensor.mean()),
        median=float(tensor.median()),
        p90=float(torch.quantile(tensor, 0.9)),
    )


def flat_indices_to_xy(indices: Tensor, width: int) -> Tensor:
    if width <= 0:
        raise ValueError('feature-map width must be positive')
    return torch.stack((indices.remainder(width),
                        torch.div(indices, width, rounding_mode='floor')),
                       dim=-1).float()


def select_queries(localized_heatmap: Tensor,
                   num_proposals: int) -> Dict[str, Tensor]:
    if localized_heatmap.ndim != 4 or localized_heatmap.shape[0] != 1:
        raise ValueError('localized_heatmap must have shape [1,C,H,W]')
    _, _, height, width = localized_heatmap.shape
    spatial_size = height * width
    flat = localized_heatmap.flatten(2)
    top = flat.reshape(1, -1).argsort(
        dim=-1, descending=True)[..., :num_proposals]
    labels = top // spatial_size
    indices = top % spatial_size
    scores = flat.gather(2, indices[:, None].expand(-1, flat.shape[1], -1))
    return dict(
        labels=labels,
        indices=indices,
        xy=flat_indices_to_xy(indices, width),
        class_scores=scores,
    )


def query_metrics(query_xy: Tensor, query_labels: Tensor, gt_xy: Tensor,
                  gt_labels: Tensor) -> dict:
    nearest = gt_xy.new_full((len(gt_xy),), float('inf'))
    no_same_class = torch.ones(len(gt_xy), dtype=torch.bool,
                               device=gt_xy.device)
    for gt_index, label in enumerate(gt_labels):
        mask = query_labels == label
        if mask.any():
            no_same_class[gt_index] = False
            nearest[gt_index] = torch.linalg.vector_norm(
                query_xy[mask] - gt_xy[gt_index], dim=1).min()
    return dict(
        nearest_distance=nearest,
        recall={radius: nearest <= radius for radius in RECALL_RADII},
        no_same_class=no_same_class,
    )


def absolute_yaw_error(pred_yaw: Tensor, gt_yaw: Tensor) -> Tensor:
    delta = pred_yaw - gt_yaw
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()
```

- [ ] **Step 4: 运行 Task 2 测试**

Run:

```bash
python -m pytest -q \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
```

Expected: `8 passed`。

- [ ] **Step 5: 提交 Task 2**

```bash
git add projects/TransFusionKITTI/tools/diagnose_stage1.py \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
git commit -m "feat: add Stage 1 query diagnostic primitives"
```

### Task 3: Production Hungarian 匹配与框误差记录

**Files:**
- Modify: `projects/TransFusionKITTI/tools/diagnose_stage1.py`
- Modify: `projects/TransFusionKITTI/tests/test_diagnose_stage1.py`

- [ ] **Step 1: 写 decoder 匹配记录失败测试**

使用一个显式 `AssignResult` 和可预测的 BEV IoU 函数，验证 query/GT 索引、中心、尺寸、yaw、3D IoU 和最终分数均按 production 匹配对取值：

```python
from mmdet.models.task_modules import AssignResult


def test_build_match_record_uses_assigned_query_gt_pairs():
    pred_boxes = torch.tensor([
        [10., 1., -1., 4., 2., 1.5, 0.2],
        [20., 2., -1., 1., 1., 1.7, -0.2],
    ])
    gt_boxes = torch.tensor([
        [10.5, 1.5, -1.2, 3.5, 1.5, 1.4, 0.1],
        [20.5, 2.5, -0.8, 1.2, 0.8, 1.6, -0.1],
    ])
    gt_labels = torch.tensor([2, 0])
    query_labels = torch.tensor([2, 1])
    logits = torch.tensor([[[-8., 2.], [-8., 3.], [4., -8.]]])
    query_scores = torch.full((1, 3, 2), 0.5)
    assignment = AssignResult(
        2,
        gt_inds=torch.tensor([1, 2]),
        max_overlaps=torch.tensor([0.8, 0.4]),
        labels=gt_labels,
    )

    record = diagnose_stage1.build_match_record(
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        query_labels=query_labels,
        decoder_logits=logits,
        query_heatmap_score=query_scores,
        assignment=assignment,
        bev_iou_fn=lambda pred, gt: torch.tensor([0.7, 0.3]),
    )

    assert record['gt_labels'].tolist() == [2, 0]
    torch.testing.assert_close(record['center_abs_error'][0],
                               torch.tensor([0.5, 0.5, 0.2]))
    torch.testing.assert_close(record['dim_abs_error'][1],
                               torch.tensor([0.2, 0.2, 0.1]))
    torch.testing.assert_close(record['iou_3d'], torch.tensor([0.8, 0.4]))
    assert record['query_label_match'].tolist() == [True, False]
    assert record['strict_iou_pass'].tolist() == [True, False]
    assert record['final_gt_score'][1].item() == 0.0
```

- [ ] **Step 2: 运行匹配测试并确认失败**

Run:

```bash
python -m pytest -q \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py \
  -k build_match_record
```

Expected: FAIL with `build_match_record` missing。

- [ ] **Step 3: 实现匹配记录和正式 BEV IoU**

追加：

```python
from mmcv.ops import box_iou_rotated


def aligned_bev_iou(pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
    if len(pred_boxes) == 0:
        return pred_boxes.new_zeros((0,))
    pred_bev = pred_boxes[:, [0, 1, 3, 4, 6]]
    gt_bev = gt_boxes[:, [0, 1, 3, 4, 6]]
    return box_iou_rotated(pred_bev, gt_bev, aligned=True)


def build_match_record(pred_boxes: Tensor, gt_boxes: Tensor,
                       gt_labels: Tensor, query_labels: Tensor,
                       decoder_logits: Tensor, query_heatmap_score: Tensor,
                       assignment, bev_iou_fn=aligned_bev_iou) -> dict:
    query_indices = torch.where(assignment.gt_inds > 0)[0]
    gt_indices = assignment.gt_inds[query_indices] - 1
    matched_pred = pred_boxes[query_indices]
    matched_gt = gt_boxes[gt_indices]
    matched_labels = gt_labels[gt_indices]
    label_match = query_labels[query_indices] == matched_labels
    decoder_score = decoder_logits[0, matched_labels,
                                   query_indices].sigmoid()
    query_score = query_heatmap_score[0, matched_labels, query_indices]
    final_score = decoder_score * query_score * label_match.float()
    thresholds = matched_pred.new_tensor([
        STRICT_IOU[CLASS_NAMES[int(label)]] for label in matched_labels
    ])
    iou_3d = assignment.max_overlaps[query_indices]
    return dict(
        gt_labels=matched_labels,
        query_indices=query_indices,
        gt_indices=gt_indices,
        center_abs_error=(matched_pred[:, :3] - matched_gt[:, :3]).abs(),
        dim_abs_error=(matched_pred[:, 3:6] - matched_gt[:, 3:6]).abs(),
        yaw_abs_error=absolute_yaw_error(matched_pred[:, 6],
                                         matched_gt[:, 6]),
        bev_iou=bev_iou_fn(matched_pred, matched_gt),
        iou_3d=iou_3d,
        strict_iou_pass=iou_3d >= thresholds,
        query_label_match=label_match,
        decoder_gt_score=decoder_score,
        final_gt_score=final_score,
    )
```

- [ ] **Step 4: 运行 Task 3 测试**

Run:

```bash
python -m pytest -q \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
```

Expected: `9 passed`。

- [ ] **Step 5: 提交 Task 3**

```bash
git add projects/TransFusionKITTI/tools/diagnose_stage1.py \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
git commit -m "feat: record production Stage 1 matches"
```

### Task 4: 单样本真实数据流与严格运行时校验

**Files:**
- Modify: `projects/TransFusionKITTI/tools/diagnose_stage1.py`
- Modify: `projects/TransFusionKITTI/tests/test_diagnose_stage1.py`

- [ ] **Step 1: 写输出解包、query 一致性和缺失 checkpoint 测试**

追加：

```python
def test_unwrap_prediction_requires_expected_single_level_shape():
    prediction = {'heatmap': torch.zeros(1, 3, 2)}
    assert diagnose_stage1.unwrap_prediction(([prediction],)) is prediction
    with pytest.raises(RuntimeError, match='single-level'):
        diagnose_stage1.unwrap_prediction(([],))


def test_verify_query_reconstruction_rejects_coordinate_drift():
    selected = dict(
        labels=torch.tensor([[0, 1]]),
        class_scores=torch.tensor([[[0.8, 0.2], [0.1, 0.7]]]),
    )
    prediction = dict(
        query_labels=torch.tensor([[1, 0]]),
        query_heatmap_score=selected['class_scores'],
    )
    with pytest.raises(RuntimeError, match='query labels'):
        diagnose_stage1.verify_query_reconstruction(selected, prediction)


def test_validate_paths_rejects_missing_checkpoint(tmp_path: Path):
    config = tmp_path / 'config.py'
    config.write_text('model = dict()', encoding='utf-8')
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        diagnose_stage1.validate_paths(config, tmp_path / 'missing.pth')


def test_validate_feature_map_shape_rejects_config_drift():
    train_cfg = dict(grid_size=[1408, 1600, 40], out_size_factor=8)
    diagnose_stage1.validate_feature_map_shape((200, 176), train_cfg)
    with pytest.raises(RuntimeError, match='feature map shape'):
        diagnose_stage1.validate_feature_map_shape((176, 200), train_cfg)
```

- [ ] **Step 2: 运行新增测试并确认失败**

Run:

```bash
python -m pytest -q \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py \
  -k "unwrap_prediction or verify_query or validate_paths"
```

Expected: FAIL，指出三个 helper 尚不存在。

- [ ] **Step 3: 实现严格 helper 和 GT 网格坐标**

追加：

```python
def validate_paths(config: Path, checkpoint: Path) -> None:
    if not config.is_file():
        raise FileNotFoundError(f'config does not exist: {config}')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint does not exist: {checkpoint}')


def unwrap_prediction(raw_output) -> dict:
    if (not isinstance(raw_output, tuple) or len(raw_output) != 1 or
            not isinstance(raw_output[0], list) or len(raw_output[0]) != 1):
        raise RuntimeError('expected a single-level TransFusion prediction')
    return raw_output[0][0]


def verify_query_reconstruction(selected: dict, prediction: dict) -> None:
    if not torch.equal(selected['labels'], prediction['query_labels']):
        raise RuntimeError('reconstructed query labels differ from production')
    torch.testing.assert_close(
        selected['class_scores'],
        prediction['query_heatmap_score'],
        atol=1e-6,
        rtol=1e-5,
        msg='reconstructed query scores differ from production',
    )


def gt_centers_to_cells(gt_boxes, train_cfg) -> Tensor:
    centers = gt_boxes.gravity_center[:, :2]
    pc_range = centers.new_tensor(train_cfg['point_cloud_range'][:2])
    voxel_size = centers.new_tensor(train_cfg['voxel_size'][:2])
    return (centers - pc_range) / voxel_size / train_cfg['out_size_factor']


def validate_feature_map_shape(actual_shape, train_cfg) -> None:
    expected_shape = (
        int(train_cfg['grid_size'][1] // train_cfg['out_size_factor']),
        int(train_cfg['grid_size'][0] // train_cfg['out_size_factor']),
    )
    if tuple(actual_shape) != expected_shape:
        raise RuntimeError(
            f'feature map shape {tuple(actual_shape)} does not match '
            f'config-derived shape {expected_shape}')
```

- [ ] **Step 4: 实现单样本 production 诊断**

该函数只接收预处理后的 batch；它重建 production local-max query、调用 production bbox coder 与 Hungarian assigner、再调用正式 `predict_by_feat()` 统计最终预测：

```python
from mmengine.structures import InstanceData


def diagnose_batch(model, batch: dict) -> dict:
    inputs = batch['inputs']
    data_samples = batch['data_samples']
    if len(data_samples) != 1:
        raise RuntimeError('diagnostics require batch_size=1')
    sample = data_samples[0]
    gt = sample.gt_instances_3d
    gt_labels = gt.labels_3d
    metas = [sample.metainfo]

    raw_output = model(inputs, data_samples, mode='tensor')
    prediction = unwrap_prediction(raw_output)
    required = {
        'dense_heatmap', 'heatmap', 'center', 'height', 'dim', 'rot',
        'query_labels', 'query_heatmap_score'
    }
    missing = required.difference(prediction)
    if missing:
        raise RuntimeError(f'prediction is missing keys: {sorted(missing)}')
    for name in required:
        _require_finite(name, prediction[name])
    prediction_device = prediction['center'].device
    gt_labels = gt_labels.to(prediction_device)
    gt_boxes = gt.bboxes_3d.tensor.to(prediction_device)

    head = model.bbox_head
    dense_probability = prediction['dense_heatmap'].sigmoid()
    validate_feature_map_shape(dense_probability.shape[-2:], head.train_cfg)
    localized = head._local_max_heatmap(dense_probability)
    selected = select_queries(localized, head.num_proposals)
    verify_query_reconstruction(selected, prediction)

    gt_xy = gt_centers_to_cells(gt.bboxes_3d, head.train_cfg)
    height, width = dense_probability.shape[-2:]
    if ((gt_xy[:, 0] < 0).any() or (gt_xy[:, 0] >= width).any() or
            (gt_xy[:, 1] < 0).any() or (gt_xy[:, 1] >= height).any()):
        raise RuntimeError('filtered GT center is outside the feature map')
    gt_int = gt_xy.to(torch.long)
    dense_gt_score = dense_probability[
        0, gt_labels, gt_int[:, 1], gt_int[:, 0]]
    query_record = query_metrics(
        selected['xy'][0], selected['labels'][0], gt_xy, gt_labels)

    num_proposals = head.num_proposals
    decoder_logits = prediction['heatmap'][..., -num_proposals:]
    decoded = head.bbox_coder.decode(
        decoder_logits.detach(),
        prediction['rot'][..., -num_proposals:].detach(),
        prediction['dim'][..., -num_proposals:].detach(),
        prediction['center'][..., -num_proposals:].detach(),
        prediction['height'][..., -num_proposals:].detach(),
        filter=False,
    )[0]['bboxes']
    pred_instances = InstanceData(
        bboxes=decoded,
        scores=decoder_logits[0].transpose(0, 1),
    )
    gt_instances = InstanceData(bboxes=gt_boxes, labels=gt_labels)
    assignment = head.bbox_assigner.assign(
        pred_instances, gt_instances, head.train_cfg)
    match_record = build_match_record(
        decoded,
        gt_boxes,
        gt_labels,
        prediction['query_labels'][0],
        decoder_logits,
        prediction['query_heatmap_score'],
        assignment,
    )
    final_prediction = head.predict_by_feat(raw_output, metas)[0]
    return dict(
        gt_labels=gt_labels.detach().cpu(),
        dense_gt_score=dense_gt_score.detach().cpu(),
        nearest_query_distance=query_record['nearest_distance'].detach().cpu(),
        query_recall={
            radius: value.detach().cpu()
            for radius, value in query_record['recall'].items()
        },
        no_same_class_query=query_record['no_same_class'].detach().cpu(),
        query_labels=selected['labels'][0].detach().cpu(),
        match={key: (value.detach().cpu()
                     if isinstance(value, Tensor) else value)
               for key, value in match_record.items()},
        prediction_labels=final_prediction.labels_3d.detach().cpu(),
        prediction_scores=final_prediction.scores_3d.detach().cpu(),
        dense_shape=tuple(dense_probability.shape[-2:]),
    )
```

运行循环会在调用此函数前跳过空 GT 样本并计数；因此该函数只处理至少包含一个有效 GT 的单样本 batch。

- [ ] **Step 5: 运行 Task 4 测试和项目相关测试**

Run:

```bash
python -m pytest -q \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py \
  projects/TransFusionKITTI/tests/test_transfusion_head.py \
  projects/TransFusionKITTI/tests/test_matcher.py \
  projects/TransFusionKITTI/tests/test_bbox_coder.py
```

Expected: all selected tests pass。

- [ ] **Step 6: 提交 Task 4**

```bash
git add projects/TransFusionKITTI/tools/diagnose_stage1.py \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
git commit -m "feat: diagnose Stage 1 production data flow"
```

### Task 5: 聚合 JSON、终端摘要与主运行入口

**Files:**
- Modify: `projects/TransFusionKITTI/tools/diagnose_stage1.py`
- Modify: `projects/TransFusionKITTI/tests/test_diagnose_stage1.py`

- [ ] **Step 1: 写分组聚合和 JSON 失败测试**

追加以下测试，确保空类别输出 `None` 而非 NaN，并且最终结构可由标准库序列化：

```python
import json


def test_grouped_summary_keeps_empty_class_json_safe():
    result = diagnose_stage1.grouped_summary(
        values=torch.tensor([0.2, 0.8]),
        labels=torch.tensor([0, 2]),
        class_names=diagnose_stage1.CLASS_NAMES,
    )
    assert result['overall']['count'] == 2
    assert result['by_class']['Cyclist']['count'] == 0
    assert result['by_class']['Cyclist']['mean'] is None
    json.dumps(result, allow_nan=False)


def test_aggregate_records_has_required_top_level_sections():
    record = dict(
        gt_labels=torch.tensor([0]),
        dense_gt_score=torch.tensor([0.4]),
        nearest_query_distance=torch.tensor([1.0]),
        query_recall={radius: torch.tensor([radius >= 1.0])
                      for radius in diagnose_stage1.RECALL_RADII},
        no_same_class_query=torch.tensor([False]),
        query_labels=torch.tensor([0, 2]),
        match=dict(
            gt_labels=torch.tensor([0]),
            center_abs_error=torch.tensor([[1., 2., 3.]]),
            dim_abs_error=torch.tensor([[0.1, 0.2, 0.3]]),
            yaw_abs_error=torch.tensor([0.1]),
            bev_iou=torch.tensor([0.4]),
            iou_3d=torch.tensor([0.3]),
            strict_iou_pass=torch.tensor([False]),
            query_label_match=torch.tensor([True]),
            decoder_gt_score=torch.tensor([0.5]),
            final_gt_score=torch.tensor([0.25]),
        ),
        prediction_labels=torch.tensor([0, 2]),
        prediction_scores=torch.tensor([0.4, 0.1]),
    )
    result = diagnose_stage1.aggregate_records([record])
    assert set(result) == {
        'ground_truth', 'dense_queries', 'decoder_matches', 'scores'
    }
    assert result['ground_truth']['count']['overall'] == 1
    json.dumps(result, allow_nan=False)
```

- [ ] **Step 2: 运行聚合测试并确认失败**

Run:

```bash
python -m pytest -q \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py \
  -k "grouped_summary or aggregate_records"
```

Expected: FAIL，指出聚合函数缺失。

- [ ] **Step 3: 实现分组和跨样本聚合**

实现以下接口；`aggregate_records()` 对每个字段先 `torch.cat`，再调用 `grouped_summary()`：

```python
def grouped_summary(values: Tensor, labels: Tensor,
                    class_names=CLASS_NAMES) -> dict:
    values = values.reshape(-1).tolist()
    labels = labels.reshape(-1).tolist()
    if len(values) != len(labels):
        raise ValueError('values and labels must have equal length')
    return dict(
        overall=numeric_summary(values),
        by_class={
            name: numeric_summary(
                value for value, label in zip(values, labels)
                if label == class_index)
            for class_index, name in enumerate(class_names)
        },
    )


def _concat(records, key, nested=None):
    values = [record[nested][key] if nested else record[key]
              for record in records]
    return torch.cat(values) if values else torch.empty(0)


def _counts(labels: Tensor) -> dict:
    return dict(
        overall=int(labels.numel()),
        by_class={name: int((labels == index).sum())
                  for index, name in enumerate(CLASS_NAMES)},
    )


def aggregate_records(records: list) -> dict:
    gt_labels = _concat(records, 'gt_labels')
    match_labels = _concat(records, 'gt_labels', nested='match')
    prediction_labels = _concat(records, 'prediction_labels')
    nearest = _concat(records, 'nearest_query_distance')
    nearest_mask = torch.isfinite(nearest)
    decoder = {}
    for name in ('center_abs_error', 'dim_abs_error'):
        values = _concat(records, name, nested='match')
        axis_names = ('x', 'y', 'z') if name == 'center_abs_error' else (
            'dx', 'dy', 'dz')
        decoder[name] = {
            axis: grouped_summary(values[:, axis_index], match_labels)
            for axis_index, axis in enumerate(axis_names)
        }
    for name in ('yaw_abs_error', 'bev_iou', 'iou_3d',
                 'strict_iou_pass', 'query_label_match'):
        decoder[name] = grouped_summary(
            _concat(records, name, nested='match').float(), match_labels)
    return dict(
        ground_truth=dict(count=_counts(gt_labels)),
        dense_queries=dict(
            gt_center_score=grouped_summary(
                _concat(records, 'dense_gt_score'), gt_labels),
            nearest_same_class_distance_cells=grouped_summary(
                nearest[nearest_mask], gt_labels[nearest_mask]),
            same_class_recall_cells={
                str(int(radius)): grouped_summary(
                    torch.cat([record['query_recall'][radius]
                               for record in records]).float(), gt_labels)
                for radius in RECALL_RADII
            },
            no_same_class_query=grouped_summary(
                _concat(records, 'no_same_class_query').float(), gt_labels),
            query_class_count=_counts(_concat(records, 'query_labels')),
        ),
        decoder_matches=decoder,
        scores=dict(
            decoder_gt_class_score=grouped_summary(
                _concat(records, 'decoder_gt_score', nested='match'),
                match_labels),
            final_gt_class_score=grouped_summary(
                _concat(records, 'final_gt_score', nested='match'),
                match_labels),
            final_prediction_count=_counts(prediction_labels),
            final_prediction_score=grouped_summary(
                _concat(records, 'prediction_scores'), prediction_labels),
        ),
    )
```

`nearest_query_distance` 为 `inf` 的 GT 只计入 `no_same_class_query`，不会送入 `numeric_summary()`；最近距离统计的 `count` 因此表示“存在同类 query 的 GT 数量”。

- [ ] **Step 4: 实现模型装配、循环、metadata、JSON 与终端摘要**

追加 `run()` 和 `main()`。真实 checkpoint 必须 `strict=True`，只处理固定顺序的前 `max_samples` 个非空验证样本：

```python
def run(args) -> dict:
    import json

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint
    from mmengine.utils import import_modules_from_strings

    from mmdet3d.registry import MODELS

    if args.max_samples <= 0:
        raise ValueError('--max-samples must be positive')
    validate_paths(args.config, args.checkpoint)
    cfg = Config.fromfile(str(args.config))
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    validate_stage1_config(cfg)

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = MODELS.build(cfg.model)
    model.init_weights()
    load_checkpoint(model, str(args.checkpoint), map_location='cpu',
                    strict=True)
    model.to(device).eval()
    dataloader = Runner.build_dataloader(
        build_diagnostic_dataloader_cfg(cfg))

    records = []
    empty_samples = 0
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = model.data_preprocessor(raw_batch, training=False)
            if len(batch['data_samples'][0].gt_instances_3d.labels_3d) == 0:
                empty_samples += 1
                continue
            records.append(diagnose_batch(model, batch))
            if len(records) == args.max_samples:
                break
    if not records:
        raise RuntimeError('diagnostic subset contains no valid GT')

    aggregated = aggregate_records(records)
    dense_shapes = {tuple(record['dense_shape']) for record in records}
    if len(dense_shapes) != 1:
        raise RuntimeError(f'inconsistent feature map shapes: {dense_shapes}')
    dense_shape = next(iter(dense_shapes))
    result = dict(
        metadata=dict(
            config=str(args.config.resolve()),
            checkpoint=str(args.checkpoint.resolve()),
            device=str(device),
            processed_samples=len(records),
            skipped_empty_samples=empty_samples,
            class_names=list(CLASS_NAMES),
            bev_feature_shape=list(dense_shape),
            voxel_size=list(cfg.voxel_size),
            out_size_factor=int(cfg.out_size_factor),
            point_cloud_range=list(cfg.point_cloud_range),
        ),
        **aggregated,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print_summary(result, args.output)
    return result


def print_summary(result: dict, output: Path) -> None:
    dense = result['dense_queries']
    matches = result['decoder_matches']
    scores = result['scores']
    print(f"STAGE1_DIAGNOSTICS_OK samples="
          f"{result['metadata']['processed_samples']}")
    print('dense_gt_score:', dense['gt_center_score'])
    print('query_recall_cells:', dense['same_class_recall_cells'])
    print('matched_iou_3d:', matches['iou_3d'])
    print('query_label_match:', matches['query_label_match'])
    print('decoder_gt_score:', scores['decoder_gt_class_score'])
    print('final_gt_score:', scores['final_gt_class_score'])
    print(f'JSON: {output}')


def main() -> None:
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
```

`diagnose_batch()` 的返回值同步增加 `dense_shape=tuple(dense_probability.shape[-2:])`。如果不同样本 shape 不一致，`aggregate_records()` 前立即报错。输出目录创建和 JSON 写入异常不得捕获为成功信息。

- [ ] **Step 5: 运行诊断脚本单测**

Run:

```bash
python -m pytest -q \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
```

Expected: all tests pass，且没有 warning 被转成 NaN/Inf 输出。

- [ ] **Step 6: 提交 Task 5**

```bash
git add projects/TransFusionKITTI/tools/diagnose_stage1.py \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
git commit -m "feat: add read-only Stage 1 diagnostic report"
```

### Task 6: 完整本地验收与 H800 服务器命令

**Files:**
- Verify only: `projects/TransFusionKITTI/tools/diagnose_stage1.py`
- Verify only: `projects/TransFusionKITTI/tests/test_diagnose_stage1.py`

- [ ] **Step 1: 运行 TransFusionKITTI 完整测试**

Run:

```bash
python -m pytest -q projects/TransFusionKITTI/tests
```

Expected: all project tests pass。依赖完整的 H800 环境不得出现 skip；本地 Windows 若因 CUDA/MMCV 依赖产生既有 skip，要逐项记录，不能宣称服务器运行已验证。

- [ ] **Step 2: 运行编译和格式检查**

Run:

```bash
python -m compileall -q projects/TransFusionKITTI
git diff --check
```

Expected: both commands exit 0。

- [ ] **Step 3: 检查只读边界**

Run:

```bash
git diff 2dd5d7c6 -- \
  projects/TransFusionKITTI/transfusion_kitti \
  projects/TransFusionKITTI/configs
```

Expected: no output，证明实现未修改模型和配置。

- [ ] **Step 4: 提交实现计划执行过程中的必要修正**

只有前述测试或静态检查确实要求修正时才创建该提交；没有修正则跳过：

```bash
git add projects/TransFusionKITTI/tools/diagnose_stage1.py \
  projects/TransFusionKITTI/tests/test_diagnose_stage1.py
git commit -m "test: harden Stage 1 diagnostic tool"
```

- [ ] **Step 5: 用户复制必要文件后在 H800 运行一次真实诊断**

只需把新脚本复制到服务器同一路径；测试文件不是服务器运行必需。仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=2 python \
  projects/TransFusionKITTI/tools/diagnose_stage1.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --checkpoint "work_dirs/transfusion_l_kitti_formal_run1_xyfix/best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_epoch_5.pth" \
  --max-samples 64 \
  --output work_dirs/transfusion_l_kitti_formal_run1_xyfix/stage1_diagnostics.json
```

Expected:

```text
STAGE1_DIAGNOSTICS_OK samples=64
...
JSON: work_dirs/transfusion_l_kitti_formal_run1_xyfix/stage1_diagnostics.json
```

该命令不启动训练、不需要 Stage 0 checkpoint、不覆盖现有 Stage 1 checkpoint。真实数值输出在服务器执行前保持“待验证”；拿到终端摘要和 JSON 后，才根据四段指标判断下一处修复位置。
