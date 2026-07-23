# TransFusion KITTI Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 MMDetection3D 1.4 中实现保留原版 query-level 相机-LiDAR融合逻辑的 TransFusion，并完成 KITTI 三类别训练、推理和 AP_R40 评估适配。

**Architecture:** 在 `projects/TransFusionKITTI` 中独立注册现代 detector、head、bbox coder、匹配器和 KITTI投影工具。LiDAR query 主路径以 MMDetection3D 1.4 中已迁移的 TransFusionHead 为参考，图像分支严格移植原版 TransFusion 的垂直压缩图像引导、query 投影、Gaussian attention mask 和 query-image cross-attention，不引入 BEVFusion 的 LSS/BEV图像融合。

**Tech Stack:** Python 3.8、PyTorch 2.1.2、CUDA 11.8、MMCV 2.1.0、MMEngine 0.10.x、MMDetection 3.2.0、MMDetection3D 1.4.0、spconv 2.x、pytest。

---

## 执行状态（2026-07-22）

- Tasks 1-14：源码、配置、工具、测试与操作文档已实现。
- Task 15：格式、语法、配置解析和纯 Python 测试已通过；由于本机缺少
  PyTorch/MMCV，模型注册、构建及前后向测试仍待目标环境验证。
- Task 16：待在 H800、完整依赖和 KITTI 数据环境执行，不得提前表述为
  “已跑通”或“已验证有效”。

## File Map

```text
projects/TransFusionKITTI/
├── README.md                                      # 环境、数据、三阶段训练和测试命令
├── configs/
│   ├── transfusion_l_kitti.py                    # 阶段 1，LiDAR-only
│   ├── transfusion_lc_kitti.py                   # 阶段 2，query-level 融合
│   └── r50_fpn_kitti_2d.py                       # 阶段 0，KITTI 2D预训练
├── transfusion_kitti/
│   ├── __init__.py                               # 项目统一注册入口
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transfusion_detector.py               # 点云/图像特征提取、冻结和 head 调用
│   │   ├── transfusion_head.py                   # LiDAR query 和原版图像融合
│   │   ├── transformer.py                        # TransFusion decoder 和位置编码
│   │   ├── projection.py                         # KITTI query/3D框投影与有效性 mask
│   │   └── task_modules/
│   │       ├── __init__.py
│   │       ├── transfusion_bbox_coder.py         # 无速度 8 维编码/解码
│   │       ├── hungarian_assigner_3d.py          # 现代 InstanceData 匹配器
│   │       └── match_costs.py                    # BEV L1 和 3D IoU cost
│   └── datasets/
│       ├── __init__.py
│       └── transforms.py                         # KITTI校准验证
├── tools/
│   ├── convert_kitti_2d_to_coco.py               # label_2 -> COCO JSON
│   ├── merge_pretrained_weights.py               # 阶段 0/1 -> 阶段 2 checkpoint
│   └── smoke_test.py                              # 单样本 loss/predict/backward
└── tests/
    ├── test_bbox_coder.py
    ├── test_matcher.py
    ├── test_projection.py
    ├── test_transfusion_head.py
    ├── test_transfusion_detector.py
    ├── test_transforms.py
    ├── test_kitti_2d_converter.py
    ├── test_checkpoint_merge.py
    └── test_configs.py
```

原版只作为行为参考：

```text
../TransFusion/TransFusion/mmdet3d/models/detectors/transfusion.py
../TransFusion/TransFusion/mmdet3d/models/dense_heads/transfusion_head.py
../TransFusion/TransFusion/mmdet3d/core/bbox/coders/transfusion_bbox_coder.py
```

现代 LiDAR query 参考：

```text
projects/BEVFusion/bevfusion/transfusion_head.py
projects/BEVFusion/bevfusion/transformer.py
projects/BEVFusion/bevfusion/utils.py
```

### Task 1: Project Scaffold and Registration

**Files:**
- Create: `projects/TransFusionKITTI/transfusion_kitti/__init__.py`
- Create: `projects/TransFusionKITTI/transfusion_kitti/models/__init__.py`
- Create: `projects/TransFusionKITTI/transfusion_kitti/models/task_modules/__init__.py`
- Create: `projects/TransFusionKITTI/transfusion_kitti/datasets/__init__.py`
- Create: `projects/TransFusionKITTI/tests/test_registration.py`

- [ ] **Step 1: Write the failing registration test**

```python
from importlib import import_module


def test_project_package_imports():
    module = import_module(
        'projects.TransFusionKITTI.transfusion_kitti')
    assert module.__all__ == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests/test_registration.py
```

Expected: FAIL with `ModuleNotFoundError: projects.TransFusionKITTI`.

- [ ] **Step 3: Create the package initializers**

Initial root initializer:

```python
"""Modern TransFusion components for KITTI."""

__all__ = []
```

The other three initializers contain the same module docstring and `__all__ = []`. Components are added to these exports in the task that creates them.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests/test_registration.py
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/TransFusionKITTI
git commit -m "build: scaffold TransFusion KITTI project"
```

### Task 2: Velocity-Free KITTI BBox Coder

**Files:**
- Create: `projects/TransFusionKITTI/transfusion_kitti/models/task_modules/transfusion_bbox_coder.py`
- Modify: `projects/TransFusionKITTI/transfusion_kitti/models/task_modules/__init__.py`
- Modify: `projects/TransFusionKITTI/transfusion_kitti/models/__init__.py`
- Modify: `projects/TransFusionKITTI/transfusion_kitti/__init__.py`
- Create: `projects/TransFusionKITTI/tests/test_bbox_coder.py`

- [ ] **Step 1: Write encode/decode and non-mutation tests**

```python
import torch

from projects.TransFusionKITTI.transfusion_kitti.models.task_modules import (
    TransFusionKITTIBBoxCoder)


def make_coder():
    return TransFusionKITTIBBoxCoder(
        pc_range=[0.0, -40.0],
        out_size_factor=8,
        voxel_size=[0.05, 0.05],
        post_center_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
        score_threshold=0.0,
        code_size=8)


def test_bbox_coder_round_trip_without_velocity():
    coder = make_coder()
    boxes = torch.tensor([[12.0, -2.0, -1.5, 3.9, 1.6, 1.5, 0.25]])
    encoded = coder.encode(boxes)
    heatmap = torch.tensor([[[0.9]]])
    decoded = coder.decode(
        heatmap=heatmap,
        rot=encoded[:, 6:8].T.unsqueeze(0),
        dim=encoded[:, 3:6].T.unsqueeze(0),
        center=encoded[:, 0:2].T.unsqueeze(0),
        height=encoded[:, 2:3].T.unsqueeze(0),
        filter=False)[0]['bboxes']
    torch.testing.assert_close(decoded[0], boxes[0], atol=1e-5, rtol=1e-5)


def test_bbox_coder_decode_does_not_mutate_inputs():
    coder = make_coder()
    center = torch.tensor([[[30.0], [800.0]]])
    dim = torch.zeros(1, 3, 1)
    center_before, dim_before = center.clone(), dim.clone()
    coder.decode(
        torch.tensor([[[0.9]]]), torch.tensor([[[0.0], [1.0]]]),
        dim, center, torch.zeros(1, 1, 1), filter=False)
    torch.testing.assert_close(center, center_before)
    torch.testing.assert_close(dim, dim_before)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest -q projects/TransFusionKITTI/tests/test_bbox_coder.py
```

Expected: FAIL because `TransFusionKITTIBBoxCoder` is not defined.

- [ ] **Step 3: Implement the coder**

Implement `TransFusionKITTIBBoxCoder(BaseBBoxCoder)` with `@TASK_UTILS.register_module()`. `encode()` creates an 8-column target tensor; `decode()` begins with clones and accepts no velocity argument:

```python
center = center.clone()
dim = dim.clone().exp()
center[:, 0] = (center[:, 0] * self.out_size_factor
                * self.voxel_size[0] + self.pc_range[0])
center[:, 1] = (center[:, 1] * self.out_size_factor
                * self.voxel_size[1] + self.pc_range[1])
height = height - dim[:, 2:3] * 0.5
yaw = torch.atan2(rot[:, 0:1], rot[:, 1:2])
boxes = torch.cat([center, height, dim, yaw], dim=1).permute(0, 2, 1)
```

Create the range tensor per call without mutating `self.post_center_range`:

```python
post_range = boxes.new_tensor(self.post_center_range)
range_mask = ((boxes[..., :3] >= post_range[:3]).all(-1)
              & (boxes[..., :3] <= post_range[3:]).all(-1))
```

Export the class from all package initializers.

- [ ] **Step 4: Run focused tests**

```bash
pytest -q projects/TransFusionKITTI/tests/test_bbox_coder.py
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/TransFusionKITTI
git commit -m "feat: add KITTI TransFusion bbox coder"
```

### Task 3: Hungarian Matching Utilities

**Files:**
- Create: `projects/TransFusionKITTI/transfusion_kitti/models/task_modules/match_costs.py`
- Create: `projects/TransFusionKITTI/transfusion_kitti/models/task_modules/hungarian_assigner_3d.py`
- Modify: `projects/TransFusionKITTI/transfusion_kitti/models/task_modules/__init__.py`
- Create: `projects/TransFusionKITTI/tests/test_matcher.py`

- [ ] **Step 1: Write deterministic matching tests**

```python
import torch
from mmengine.structures import InstanceData

from projects.TransFusionKITTI.transfusion_kitti.models.task_modules import (
    TransFusionKITTIHungarianAssigner3D)


def test_hungarian_assigner_matches_same_class_nearest_boxes():
    assigner = TransFusionKITTIHungarianAssigner3D(
        iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
        cls_cost=dict(
            type='mmdet.FocalLossCost', weight=0.15, alpha=0.25, gamma=2.0),
        reg_cost=dict(type='TransFusionKITTIBBoxBEVL1Cost', weight=0.25),
        iou_cost=dict(type='TransFusionKITTIIoU3DCost', weight=0.25))
    pred = InstanceData(
        bboxes=torch.tensor([
            [10., 0., -1., 4., 2., 1.5, 0.],
            [30., 0., -1., 4., 2., 1.5, 0.],
        ]),
        scores=torch.tensor([[8., -8., -8.], [-8., 8., -8.]]))
    gt = InstanceData(
        bboxes=torch.tensor([
            [10.1, 0., -1., 4., 2., 1.5, 0.],
            [30.1, 0., -1., 4., 2., 1.5, 0.],
        ]),
        labels=torch.tensor([0, 1]))
    result = assigner.assign(
        pred, gt, dict(point_cloud_range=[0., -40., -3., 70.4, 40., 1.]))
    assert result.gt_inds.tolist() == [1, 2]
    assert result.labels.tolist() == [0, 1]
```

- [ ] **Step 2: Verify failure**

```bash
pytest -q projects/TransFusionKITTI/tests/test_matcher.py
```

Expected: FAIL because the assigner and costs are unregistered.

- [ ] **Step 3: Port modern task utilities under unique names**

Use `projects/BEVFusion/bevfusion/utils.py` lines defining `BBoxBEVL1Cost`, `IoU3DCost`, and `HungarianAssigner3D` as the source. Rename them to:

```python
TransFusionKITTIBBoxBEVL1Cost
TransFusionKITTIIoU3DCost
TransFusionKITTIHungarianAssigner3D
```

Register all three with `TASK_UTILS`. Keep the modern `InstanceData` interface and return `AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)`. Add an explicit empty-GT branch:

```python
if len(gt_instances) == 0:
    return AssignResult(
        0,
        pred_instances.bboxes.new_zeros(len(pred_instances), dtype=torch.long),
        pred_instances.bboxes.new_zeros(len(pred_instances)),
        labels=pred_instances.bboxes.new_full(
            (len(pred_instances),), -1, dtype=torch.long))
```

- [ ] **Step 4: Run matcher and coder tests**

```bash
pytest -q \
  projects/TransFusionKITTI/tests/test_matcher.py \
  projects/TransFusionKITTI/tests/test_bbox_coder.py
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/TransFusionKITTI
git commit -m "feat: add TransFusion KITTI matcher"
```

### Task 4: Modern LiDAR Query Head

**Files:**
- Create: `projects/TransFusionKITTI/transfusion_kitti/models/transformer.py`
- Create: `projects/TransFusionKITTI/transfusion_kitti/models/transfusion_head.py`
- Modify: `projects/TransFusionKITTI/transfusion_kitti/models/__init__.py`
- Create: `projects/TransFusionKITTI/tests/test_transfusion_head.py`

- [ ] **Step 1: Write a synthetic LiDAR forward test**

```python
import torch
from mmengine.config import ConfigDict

from projects.TransFusionKITTI.transfusion_kitti.models import (
    TransFusionKITTIHead)


def make_head(fuse_img=False):
    return TransFusionKITTIHead(
        fuse_img=fuse_img,
        num_proposals=20,
        auxiliary=True,
        in_channels=512,
        hidden_channel=64,
        num_classes=3,
        nms_kernel_size=3,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransFusionKITTITransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=64, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=64, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=64, feedforward_channels=128, num_fcs=2,
                ffn_drop=0.0, act_cfg=dict(type='ReLU', inplace=True)),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=64)),
        train_cfg=None,
        test_cfg=ConfigDict(
            dataset='KITTI', grid_size=[160, 160, 40], out_size_factor=8,
            voxel_size=[0.05, 0.05], pc_range=[0.0, -40.0], nms_type=None),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]),
        bbox_coder=dict(
            type='TransFusionKITTIBBoxCoder', pc_range=[0.0, -40.0],
            post_center_range=[0., -40., -3., 70.4, 40., 1.],
            score_threshold=0.0, out_size_factor=8,
            voxel_size=[0.05, 0.05], code_size=8),
        loss_cls=dict(
            type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0,
            alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25))


def test_lidar_head_forward_has_no_velocity_and_keeps_batch_state_in_output():
    head = make_head(fuse_img=False).eval()
    outputs = head([torch.randn(2, 512, 20, 20)], [{}, {}])
    pred = outputs[0][0]
    assert 'vel' not in pred
    assert pred['center'].shape == (2, 2, 20)
    assert pred['query_labels'].shape == (2, 20)
    assert not hasattr(head, 'query_labels')
```

- [ ] **Step 2: Verify failure**

```bash
pytest -q projects/TransFusionKITTI/tests/test_transfusion_head.py
```

Expected: FAIL because the transformer and head do not exist.

- [ ] **Step 3: Copy the maintained LiDAR implementation**

Create `transformer.py` from the complete contents of:

```text
projects/BEVFusion/bevfusion/transformer.py
```

Rename the registered decoder to `TransFusionKITTITransformerDecoderLayer`.

Create `transfusion_head.py` from the complete `TransFusionHead` definition in:

```text
projects/BEVFusion/bevfusion/transfusion_head.py
```

Do not copy `ConvFuser`. Rename the head to `TransFusionKITTIHead`, switch task utility config names to this project's unique names, and make these exact KITTI changes:

```python
self.fuse_img = fuse_img
self.common_heads = copy.deepcopy(common_heads)
assert 'vel' not in self.common_heads
```

For local heatmap maxima, class order is `Pedestrian`, `Cyclist`, `Car`:

```python
if self.test_cfg['dataset'] == 'KITTI':
    local_max[:, 0] = heatmap[:, 0]
    local_max[:, 1] = heatmap[:, 1]
```

Return per-batch state in `pred_dict` instead of module attributes:

```python
pred_dict['query_labels'] = top_proposals_class
pred_dict['query_heatmap_score'] = heatmap.gather(
    -1, top_proposals_index[:, None].expand(-1, self.num_classes, -1))
```

All target and prediction methods read `query_labels` from `pred_dict`.

- [ ] **Step 4: Run the head tests**

```bash
pytest -q projects/TransFusionKITTI/tests/test_transfusion_head.py
```

Expected: `1 passed`.

- [ ] **Step 5: Run formatting and commit**

```bash
python -m flake8 projects/TransFusionKITTI/transfusion_kitti/models
pytest -q projects/TransFusionKITTI/tests/test_transfusion_head.py
git add projects/TransFusionKITTI
git commit -m "feat: port LiDAR TransFusion head"
```

### Task 5: Unified Detector and Freeze Policy

**Files:**
- Create: `projects/TransFusionKITTI/transfusion_kitti/models/transfusion_detector.py`
- Modify: `projects/TransFusionKITTI/transfusion_kitti/models/__init__.py`
- Create: `projects/TransFusionKITTI/tests/test_transfusion_detector.py`

- [ ] **Step 1: Write detector construction and freeze tests**

```python
import torch
from torch import nn

from projects.TransFusionKITTI.transfusion_kitti.models import (
    TransFusionKITTIDetector)


def test_freeze_module_disables_grad_and_bn_updates():
    module = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    TransFusionKITTIDetector.freeze_module(module)
    module.train()
    TransFusionKITTIDetector.freeze_module(module)
    assert not any(p.requires_grad for p in module.parameters())
    assert not module[1].training
```

- [ ] **Step 2: Verify failure**

```bash
pytest -q projects/TransFusionKITTI/tests/test_transfusion_detector.py
```

Expected: FAIL because the detector is not defined.

- [ ] **Step 3: Implement the detector**

Subclass `VoxelNet` and use the standard `Det3DDataPreprocessor` voxel output. Constructor fields are:

```python
def __init__(self,
             fuse_img=False,
             img_backbone=None,
             img_neck=None,
             img_feature_level=0,
             freeze_img=False,
             freeze_lidar=False,
             **kwargs):
```

`extract_img_feat()` accepts `[B, C, H, W]`, runs backbone and FPN, and returns exactly one feature level. `loss()` and `predict()` call:

```python
pts_feats = self.extract_feat(batch_inputs_dict)
img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs'))
return self.bbox_head.loss(
    pts_feats, batch_data_samples, img_feats=img_feats)
```

For prediction, call
`bbox_head.predict(pts_feats, batch_data_samples, img_feats=img_feats)` and
`add_pred_to_datasample(batch_data_samples, results_list)`. Implement
`_forward(batch_inputs_dict, batch_data_samples=None)` with the same feature
arguments and call `bbox_head.forward(pts_feats, batch_data_samples,
img_feats=img_feats)`.

Freeze policy:

```python
@staticmethod
def freeze_module(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False

def train(self, mode=True):
    super().train(mode)
    if mode:
        self._apply_freeze_policy()
    return self
```

`freeze_lidar=True` freezes voxel encoder, middle encoder, backbone, neck and the LiDAR-only submodules exposed by `bbox_head.lidar_modules()`. Fusion modules remain trainable.

- [ ] **Step 4: Run detector tests**

```bash
pytest -q projects/TransFusionKITTI/tests/test_transfusion_detector.py
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/TransFusionKITTI
git commit -m "feat: add TransFusion KITTI detector"
```

### Task 6: TransFusion-L KITTI Configuration

**Files:**
- Create: `projects/TransFusionKITTI/configs/transfusion_l_kitti.py`
- Create: `projects/TransFusionKITTI/tests/test_configs.py`

- [ ] **Step 1: Write configuration contract tests**

```python
from mmengine.config import Config


def test_lidar_config_contract():
    cfg = Config.fromfile(
        'projects/TransFusionKITTI/configs/transfusion_l_kitti.py')
    assert cfg.model.type == 'TransFusionKITTIDetector'
    assert cfg.model.fuse_img is False
    assert cfg.model.bbox_head.num_classes == 3
    assert 'vel' not in cfg.model.bbox_head.common_heads
    assert cfg.model.bbox_head.bbox_coder.code_size == 8
    assert cfg.train_dataloader.dataset.dataset.metainfo['classes'] == (
        'Pedestrian', 'Cyclist', 'Car')
    assert cfg.default_hooks.checkpoint.save_best == (
        'Kitti metric/pred_instances_3d/KITTI/'
        'Overall_3D_AP40_moderate')
```

- [ ] **Step 2: Verify failure**

```bash
pytest -q projects/TransFusionKITTI/tests/test_configs.py::test_lidar_config_contract
```

Expected: FAIL because the config does not exist.

- [ ] **Step 3: Create the LiDAR config**

Start from:

```python
_base_ = [
    '../../../configs/_base_/datasets/kitti-3d-3class.py',
    '../../../configs/_base_/schedules/cyclic-40e.py',
    '../../../configs/_base_/default_runtime.py',
]
custom_imports = dict(
    imports=['projects.TransFusionKITTI.transfusion_kitti'],
    allow_failed_imports=False)

class_names = ('Pedestrian', 'Cyclist', 'Car')
point_cloud_range = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]
voxel_size = [0.05, 0.05, 0.1]
grid_size = [1408, 1600, 40]
out_size_factor = 8
```

Define `TransFusionKITTIDetector` with standard `Det3DDataPreprocessor(voxel=True)`, `HardSimpleVFE`, `SparseEncoder`, SECOND, SECONDFPN and `TransFusionKITTIHead`. Head settings are:

```python
num_proposals=200
num_classes=3
num_decoder_layers=1
common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2])
code_weights=[1.0] * 8
bbox_coder=dict(
    type='TransFusionKITTIBBoxCoder',
    pc_range=[0.0, -40.0],
    post_center_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
    score_threshold=0.0,
    out_size_factor=8,
    voxel_size=[0.05, 0.05],
    code_size=8)
```

Use the project matcher and modern losses. Override checkpoint configuration:

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1,
        save_best=(
            'Kitti metric/pred_instances_3d/KITTI/'
            'Overall_3D_AP40_moderate'),
        rule='greater'))
```

- [ ] **Step 4: Load and build the config**

```bash
python tools/misc/print_config.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py
pytest -q projects/TransFusionKITTI/tests/test_configs.py
```

Expected: config prints without unresolved registry names; tests pass.

- [ ] **Step 5: Commit**

```bash
git add projects/TransFusionKITTI
git commit -m "feat: configure KITTI TransFusion-L"
```

### Task 7: KITTI Projection and Calibration Validation

**Files:**
- Create: `projects/TransFusionKITTI/transfusion_kitti/models/projection.py`
- Create: `projects/TransFusionKITTI/transfusion_kitti/datasets/transforms.py`
- Modify: `projects/TransFusionKITTI/transfusion_kitti/datasets/__init__.py`
- Create: `projects/TransFusionKITTI/tests/test_projection.py`
- Create: `projects/TransFusionKITTI/tests/test_transforms.py`

- [ ] **Step 1: Write projection edge-case tests**

```python
import torch

from projects.TransFusionKITTI.transfusion_kitti.models.projection import (
    project_lidar_points)


def test_projection_rejects_behind_camera_and_applies_homography():
    points = torch.tensor([
        [2.0, 1.0, 10.0],
        [2.0, 1.0, -1.0],
    ])
    lidar2img = torch.tensor([
        [100.0, 0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    homography = torch.tensor([
        [2.0, 0.0, 5.0],
        [0.0, 2.0, 7.0],
        [0.0, 0.0, 1.0],
    ])
    pixels, valid = project_lidar_points(
        points, lidar2img, homography, image_shape=(100, 100))
    torch.testing.assert_close(pixels[0], torch.tensor([45.0, 27.0]))
    assert valid.tolist() == [True, False]
```

Add tests for zero valid points, exactly one valid point, and points beyond all four image boundaries.

- [ ] **Step 2: Write calibration transform tests**

```python
import numpy as np
import pytest

from projects.TransFusionKITTI.transfusion_kitti.datasets import (
    ValidateKittiCalibration)


def test_calibration_validator_reports_sample_id():
    transform = ValidateKittiCalibration()
    with pytest.raises(ValueError, match='sample 42.*lidar2img'):
        transform(dict(sample_idx=42, lidar2img=np.eye(3)))
```

- [ ] **Step 3: Verify failure**

```bash
pytest -q \
  projects/TransFusionKITTI/tests/test_projection.py \
  projects/TransFusionKITTI/tests/test_transforms.py
```

Expected: FAIL because both modules are absent.

- [ ] **Step 4: Implement projection and validation**

`project_lidar_points()` uses homogeneous multiplication but tests depth before division:

```python
points_h = torch.cat([points, points.new_ones((len(points), 1))], dim=-1)
projected = points_h @ lidar2img.T
depth_valid = projected[:, 2] > eps
safe_depth = torch.where(depth_valid, projected[:, 2], torch.ones_like(projected[:, 2]))
pixels_h = torch.cat([
    projected[:, :2] / safe_depth[:, None],
    projected.new_ones((len(points), 1))
], dim=-1)
pixels_aug = pixels_h @ homography.T
pixels = pixels_aug[:, :2] / pixels_aug[:, 2:3]
h, w = image_shape
valid = (depth_valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
         & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
return pixels, valid
```

`ValidateKittiCalibration` accepts 3x4 or 4x4 `lidar2img`, normalizes to 4x4, checks finite values, and inserts identity `homography_matrix` if absent.

- [ ] **Step 5: Run tests and commit**

```bash
pytest -q \
  projects/TransFusionKITTI/tests/test_projection.py \
  projects/TransFusionKITTI/tests/test_transforms.py
git add projects/TransFusionKITTI
git commit -m "feat: add KITTI query projection"
```

### Task 8: Original Query-Level Image Fusion

**Files:**
- Modify: `projects/TransFusionKITTI/transfusion_kitti/models/transfusion_head.py`
- Modify: `projects/TransFusionKITTI/tests/test_transfusion_head.py`

- [ ] **Step 1: Add image-fusion behavior tests**

Add tests with one visible and one invalid query. Replace the image decoder with a deterministic stub so fallback is observable:

```python
def test_invalid_image_query_falls_back_to_lidar_prediction():
    head = make_head(fuse_img=True).eval()
    lidar_prediction = {'center': torch.tensor([[[1., 2.], [3., 4.]]])}
    fusion_prediction = {'center': torch.tensor([[[9., 9.], [9., 9.]]])}
    valid = torch.tensor([[True, False]])
    result = head._fallback_invalid_queries(
        fusion_prediction, lidar_prediction, valid)
    torch.testing.assert_close(
        result['center'], torch.tensor([[[9., 2.], [9., 4.]]]))


def test_single_visible_query_is_fused():
    head = make_head(fuse_img=True).eval()
    assert head._should_run_image_attention(torch.tensor([True, False]))


def test_all_invalid_query_loss_mask_is_zero_not_nan():
    mask = torch.zeros(2, 20, dtype=torch.bool)
    weights = TransFusionKITTIHead._apply_fusion_mask(
        torch.ones(2, 20), mask)
    assert weights.sum() == 0
    assert torch.isfinite(weights).all()
```

- [ ] **Step 2: Verify failure**

```bash
pytest -q projects/TransFusionKITTI/tests/test_transfusion_head.py
```

Expected: FAIL because image modules and helpers are absent.

- [ ] **Step 3: Port the original fusion blocks**

From the original head, port these behaviors into separate methods:

```python
_build_image_fusion_modules()
_image_guided_bev_heatmap(lidar_feat_flatten, img_feat)
_project_queries(pred_dict, batch_input_metas)
_fuse_visible_queries(query_feat, image_feat, projection)
_fallback_invalid_queries(fusion_pred, lidar_pred, valid_mask)
```

For KITTI, `num_views=1`. The image branch builds:

```python
self.shared_conv_img = nn.Conv2d(in_channels_img, hidden_channel, 3, padding=1)
self.image_bev_decoder = MODELS.build(copy.deepcopy(decoder_layer))
self.image_query_decoder = MODELS.build(copy.deepcopy(decoder_layer))
self.image_heatmap_head = copy.deepcopy(self.heatmap_head)
self.fusion_prediction_head = FFN(
    hidden_channel * 2, fusion_heads, conv_cfg=conv_cfg,
    norm_cfg=norm_cfg, bias=bias)
```

Use FPN P2 (`out_size_factor_img=4`). Construct the spatial mask using projected 3D box corners; clamp `sigma` to at least one feature pixel before `log()`:

```python
sigma = torch.clamp((radius * 2 + 1) / 6.0, min=1.0)
gaussian = torch.exp(-distance.square() / (2 * sigma[:, None].square()))
attn_mask = gaussian.clamp_min(torch.finfo(gaussian.dtype).tiny).log()
```

Append `valid_query_mask` to the prediction dict. Query classification and bbox weights are multiplied by this mask only for the fusion prediction layer. Dense image heatmap loss remains active.

- [ ] **Step 4: Run all head tests**

```bash
pytest -q projects/TransFusionKITTI/tests/test_transfusion_head.py
```

Expected: all LiDAR and image-fusion tests pass.

- [ ] **Step 5: Commit**

```bash
git add projects/TransFusionKITTI
git commit -m "feat: port TransFusion query image fusion"
```

### Task 9: Multimodal Detector Path and KITTI-LC Config

**Files:**
- Modify: `projects/TransFusionKITTI/transfusion_kitti/models/transfusion_detector.py`
- Create: `projects/TransFusionKITTI/configs/transfusion_lc_kitti.py`
- Modify: `projects/TransFusionKITTI/tests/test_transfusion_detector.py`
- Modify: `projects/TransFusionKITTI/tests/test_configs.py`

- [ ] **Step 1: Write LC config and detector input tests**

```python
from mmengine.config import Config


def test_lc_config_uses_single_camera_without_geometry_augmentation():
    cfg = Config.fromfile(
        'projects/TransFusionKITTI/configs/transfusion_lc_kitti.py')
    assert cfg.model.fuse_img is True
    assert cfg.model.bbox_head.fuse_img is True
    assert cfg.model.bbox_head.num_views == 1
    pipeline_types = [item['type'] for item in cfg.train_pipeline]
    assert 'LoadImageFromFileMono3D' in pipeline_types
    assert 'ObjectSample' not in pipeline_types
    assert 'ObjectNoise' not in pipeline_types
    assert 'RandomFlip3D' not in pipeline_types
    assert 'GlobalRotScaleTrans' not in pipeline_types
```

Add a detector test asserting `[B,C,H,W]` image input produces P2 features and that `fuse_img=False` does not access `imgs`.

- [ ] **Step 2: Verify failure**

```bash
pytest -q \
  projects/TransFusionKITTI/tests/test_transfusion_detector.py \
  projects/TransFusionKITTI/tests/test_configs.py
```

Expected: FAIL because LC config and completed image path are absent.

- [ ] **Step 3: Complete detector image extraction**

Use ResNet output tuple and FPN tuple, then select P2:

```python
features = self.img_backbone(imgs.float())
features = self.img_neck(features) if self.img_neck is not None else features
return features[self.img_feature_level]
```

Normalize image shape so the head always receives `[B,1,C,H,W]` internally. Pass each sample's `lidar2img`, `homography_matrix`, `img_shape` and `pad_shape` through its metainfo.

- [ ] **Step 4: Create the LC config**

Base it on `transfusion_l_kitti.py`, then set:

```python
input_modality = dict(use_lidar=True, use_camera=True)
model = dict(
    fuse_img=True,
    freeze_img=True,
    freeze_lidar=True,
    img_feature_level=0,
    img_backbone=dict(
        type='mmdet.ResNet', depth=50, num_stages=4,
        out_indices=(0, 1, 2, 3), frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False), norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='mmdet.FPN', in_channels=[256, 512, 1024, 2048],
        out_channels=256, num_outs=5),
    bbox_head=dict(
        fuse_img=True, num_views=1, in_channels_img=256,
        out_size_factor_img=4))
```

Fusion pipeline order:

```python
LoadImageFromFileMono3D
LoadPointsFromFile(load_dim=4, use_dim=4)
LoadAnnotations3D
ValidateKittiCalibration
Resize(scale=(1280, 384), keep_ratio=True)
PointsRangeFilter
ObjectRangeFilter
PointShuffle
Pack3DDetInputs(keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
                meta_keys=('sample_idx', 'lidar2img', 'cam2img',
                           'lidar2cam', 'ori_shape', 'img_shape',
                           'pad_shape', 'scale_factor',
                           'homography_matrix'))
```

Set `train_cfg.max_epochs=6`. Keep the paper-faithful baseline in FP32. Do not
enable the modern global AMP override until the original `force_fp32` loss
boundary has been ported and verified separately.

- [ ] **Step 5: Run tests and commit**

```bash
pytest -q \
  projects/TransFusionKITTI/tests/test_transfusion_detector.py \
  projects/TransFusionKITTI/tests/test_configs.py
git add projects/TransFusionKITTI
git commit -m "feat: configure KITTI TransFusion-LC"
```

### Task 10: KITTI 2D Annotation Converter

**Files:**
- Create: `projects/TransFusionKITTI/tools/convert_kitti_2d_to_coco.py`
- Create: `projects/TransFusionKITTI/tests/test_kitti_2d_converter.py`

- [ ] **Step 1: Write a temporary-dataset conversion test**

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

from projects.TransFusionKITTI.tools.convert_kitti_2d_to_coco import convert


def test_converter_filters_dontcare_and_keeps_three_classes(tmp_path: Path):
    root = tmp_path / 'kitti'
    (root / 'training/image_2').mkdir(parents=True)
    (root / 'training/label_2').mkdir(parents=True)
    (root / 'ImageSets').mkdir()
    Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(
        root / 'training/image_2/000001.png')
    (root / 'training/label_2/000001.txt').write_text(
        'Pedestrian 0 0 0 10 20 30 60 1 1 1 1 1 1 0\n'
        'DontCare -1 -1 -10 0 0 40 40 -1 -1 -1 -1000 -1000 -1000 -10\n',
        encoding='utf-8')
    (root / 'ImageSets/train.txt').write_text('000001\n', encoding='utf-8')
    output = root / 'annotations/kitti_2d_train.json'
    convert(root, root / 'ImageSets/train.txt', output)
    data = json.loads(output.read_text(encoding='utf-8'))
    assert [c['name'] for c in data['categories']] == [
        'Pedestrian', 'Cyclist', 'Car']
    assert len(data['annotations']) == 1
    assert data['annotations'][0]['bbox'] == [10.0, 20.0, 20.0, 40.0]
```

- [ ] **Step 2: Verify failure**

```bash
pytest -q projects/TransFusionKITTI/tests/test_kitti_2d_converter.py
```

Expected: FAIL because the converter is absent.

- [ ] **Step 3: Implement the converter and CLI**

Expose:

```python
def convert(root: Path, split_file: Path, output: Path) -> None:
```

Parse all KITTI label fields `type, truncated, occluded, alpha, x1, y1, x2, y2, height, width, length, location_x, location_y, location_z, rotation_y`; discard unknown types and `DontCare`; clip boxes to image dimensions; discard zero-area boxes; write deterministic image and annotation ids. CLI:

```bash
python projects/TransFusionKITTI/tools/convert_kitti_2d_to_coco.py \
  --root data/kitti \
  --split data/kitti/ImageSets/train.txt \
  --output data/kitti/annotations/kitti_2d_train.json
```

- [ ] **Step 4: Run tests and commit**

```bash
pytest -q projects/TransFusionKITTI/tests/test_kitti_2d_converter.py
git add projects/TransFusionKITTI
git commit -m "feat: convert KITTI 2D labels to COCO"
```

### Task 11: Stage-0 ResNet-50 FPN Config

**Files:**
- Create: `projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py`
- Modify: `projects/TransFusionKITTI/tests/test_configs.py`

- [ ] **Step 1: Add the 2D config contract test**

```python
def test_2d_pretraining_config_contract():
    cfg = Config.fromfile(
        'projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py')
    assert cfg.model.type == 'FasterRCNN'
    assert cfg.model.roi_head.bbox_head.num_classes == 3
    assert cfg.train_dataloader.dataset.ann_file.endswith(
        'kitti_2d_train.json')
    assert cfg.train_dataloader.dataset.metainfo['classes'] == (
        'Pedestrian', 'Cyclist', 'Car')
```

- [ ] **Step 2: Verify failure**

```bash
pytest -q projects/TransFusionKITTI/tests/test_configs.py::test_2d_pretraining_config_contract
```

Expected: FAIL because the config is absent.

- [ ] **Step 3: Create a cross-repository MMDetection config**

Use MMEngine's package config reference:

```python
_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
default_scope = 'mmdet'
class_names = ('Pedestrian', 'Cyclist', 'Car')
data_root = 'data/kitti/'
metainfo = dict(classes=class_names)
model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))
```

Override train/val/test dataloaders with `CocoDataset`, the generated JSON files, `training/image_2/`, resize `(1280, 384)`, and standard `PackDetInputs`. Initialize from COCO Faster R-CNN R50-FPN and train all backbone/FPN weights. Use `CocoMetric(metric='bbox')`.

- [ ] **Step 4: Load and print the config**

```bash
python tools/misc/print_config.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py
pytest -q projects/TransFusionKITTI/tests/test_configs.py
```

Expected: package base resolves and config tests pass.

- [ ] **Step 5: Commit**

```bash
git add projects/TransFusionKITTI
git commit -m "feat: configure KITTI image pretraining"
```

### Task 12: Stage-0/1 Checkpoint Merge

**Files:**
- Create: `projects/TransFusionKITTI/tools/merge_pretrained_weights.py`
- Create: `projects/TransFusionKITTI/tests/test_checkpoint_merge.py`

- [ ] **Step 1: Write mapping, coverage and failure tests**

```python
import pytest
import torch

from projects.TransFusionKITTI.tools.merge_pretrained_weights import merge


def test_merge_maps_image_backbone_and_neck(tmp_path):
    lidar = tmp_path / 'lidar.pth'
    image = tmp_path / 'image.pth'
    output = tmp_path / 'merged.pth'
    torch.save({'state_dict': {'backbone.weight': torch.ones(1)}}, lidar)
    torch.save({'state_dict': {
        'backbone.conv.weight': torch.full((1,), 2.),
        'neck.fpn.weight': torch.full((1,), 3.),
        'roi_head.weight': torch.full((1,), 4.),
    }}, image)
    merged, report = merge(lidar, image, output)
    assert merged['state_dict']['img_backbone.conv.weight'].item() == 2
    assert merged['state_dict']['img_neck.fpn.weight'].item() == 3
    assert 'img_roi_head.weight' not in merged['state_dict']
    assert report['mapped_image_keys'] == 2


def test_merge_rejects_zero_image_coverage(tmp_path):
    lidar = tmp_path / 'lidar.pth'
    image = tmp_path / 'image.pth'
    output = tmp_path / 'merged.pth'
    torch.save({'state_dict': {'backbone.weight': torch.ones(1)}}, lidar)
    torch.save({'state_dict': {'roi_head.weight': torch.ones(1)}}, image)
    with pytest.raises(RuntimeError, match='no image backbone or neck keys'):
        merge(lidar, image, output)
```

- [ ] **Step 2: Verify failure**

```bash
pytest -q projects/TransFusionKITTI/tests/test_checkpoint_merge.py
```

Expected: FAIL because the merge tool is absent.

- [ ] **Step 3: Implement deterministic checkpoint merge**

Expose:

```python
def merge(lidar_path: Path, image_path: Path, output_path: Path):
```

Strip an optional `module.` prefix. Keep all LiDAR keys. Map only:

```python
backbone.* -> img_backbone.*
neck.* -> img_neck.*
```

Do not map RPN/ROI heads. Save `meta['transfusion_kitti_sources']` with input paths and SHA-256 digests. Print mapped, skipped and overwritten keys; reject zero mapped image keys and any shape collision with an existing target key.

- [ ] **Step 4: Run tests and commit**

```bash
pytest -q projects/TransFusionKITTI/tests/test_checkpoint_merge.py
git add projects/TransFusionKITTI
git commit -m "feat: merge TransFusion pretrained stages"
```

### Task 13: Single-Sample Smoke Runner

**Files:**
- Create: `projects/TransFusionKITTI/tools/smoke_test.py`
- Modify: `projects/TransFusionKITTI/tests/test_configs.py`

- [ ] **Step 1: Write CLI parsing and dry-run tests**

```python
def test_smoke_cli_accepts_lidar_and_fusion_configs():
    parser = build_parser()
    args = parser.parse_args([
        'projects/TransFusionKITTI/configs/transfusion_l_kitti.py',
        '--device', 'cpu', '--mode', 'build'])
    assert args.mode == 'build'
```

- [ ] **Step 2: Verify failure**

```bash
pytest -q projects/TransFusionKITTI/tests/test_configs.py
```

Expected: FAIL because `smoke_test.py` is absent.

- [ ] **Step 3: Implement build, loss, predict and backward modes**

The script loads the config, initializes the default scope and project imports, builds one dataset sample and model, and supports:

```text
--mode build     model construction only
--mode loss      one loss forward and finite-value assertions
--mode predict   one prediction and output schema assertions
--mode backward  one loss forward/backward and gradient-report assertions
```

For backward mode, print trainable/frozen parameter counts and fail if frozen parameters receive gradients or no fusion parameter receives a gradient.

- [ ] **Step 4: Run CPU build checks**

```bash
python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --device cpu --mode build
python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --device cpu --mode build
```

Expected: both print `MODEL_BUILD_OK`.

- [ ] **Step 5: Commit**

```bash
git add projects/TransFusionKITTI
git commit -m "test: add TransFusion KITTI smoke runner"
```

### Task 14: Project Documentation and Exact Server Commands

**Files:**
- Create: `projects/TransFusionKITTI/README.md`

- [ ] **Step 1: Write README verification expectations**

README must contain these literal command groups so they can be searched and copied independently:

```text
Environment check
KITTI data preparation
Stage 0: image pretraining
Stage 1: TransFusion-L
Checkpoint merge
Stage 2: TransFusion-LC
Evaluation
H800 smoke test
```

- [ ] **Step 2: Document data preparation**

Include:

```bash
python tools/create_data.py kitti \
  --root-path data/kitti \
  --out-dir data/kitti \
  --extra-tag kitti

python projects/TransFusionKITTI/tools/convert_kitti_2d_to_coco.py \
  --root data/kitti --split data/kitti/ImageSets/train.txt \
  --output data/kitti/annotations/kitti_2d_train.json
python projects/TransFusionKITTI/tools/convert_kitti_2d_to_coco.py \
  --root data/kitti --split data/kitti/ImageSets/val.txt \
  --output data/kitti/annotations/kitti_2d_val.json
```

- [ ] **Step 3: Document three-stage commands**

```bash
python tools/train.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py

python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py

python projects/TransFusionKITTI/tools/merge_pretrained_weights.py \
  --lidar work_dirs/transfusion_l_kitti/best_3d_moderate_mean.pth \
  --image work_dirs/r50_fpn_kitti_2d/best_coco_bbox_mAP.pth \
  --output checkpoints/transfusion_kitti_stage2_init.pth

python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --cfg-options load_from=checkpoints/transfusion_kitti_stage2_init.pth
```

Document that actual checkpoint filenames are emitted by MMEngine and the commands must use the files recorded by the relevant run, not guessed aliases.

- [ ] **Step 4: Document evaluation and smoke gates**

```bash
python tools/test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  work_dirs/transfusion_lc_kitti/best_checkpoint.pth \
  --show-dir work_dirs/transfusion_lc_kitti/visualization

python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --device cuda:0 --mode backward
```

- [ ] **Step 5: Commit**

```bash
git add projects/TransFusionKITTI/README.md
git commit -m "docs: add TransFusion KITTI workflow"
```

### Task 15: Full Local Verification

**Files:**
- Modify only files required by failures discovered in this task.

- [ ] **Step 1: Run project tests**

```bash
pytest -q projects/TransFusionKITTI/tests
```

Expected: all tests pass, with CUDA-only tests skipped when CUDA or project data is absent.

- [ ] **Step 2: Run style and static checks**

```bash
python -m flake8 projects/TransFusionKITTI
python -m compileall -q projects/TransFusionKITTI
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Verify both configs build**

```bash
python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --device cpu --mode build
python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --device cpu --mode build
```

Expected: `MODEL_BUILD_OK` twice.

- [ ] **Step 4: Review migration boundary**

Run:

```bash
rg -n "DepthLSS|LSSTransform|ConvFuser|BEVFusion\(" \
  projects/TransFusionKITTI
```

Expected: no model implementation match. Documentation may mention these names only to state they are excluded.

- [ ] **Step 5: Commit verification fixes**

```bash
git add projects/TransFusionKITTI
git commit -m "test: verify TransFusion KITTI port"
```

### Task 16: H800 Server Verification

**Files:**
- No source files unless a reproducible portability defect is found.
- Record: `work_dirs/transfusion_kitti_smoke/environment.txt`
- Record: `work_dirs/transfusion_kitti_smoke/smoke.log`

- [ ] **Step 1: Capture environment**

```bash
mkdir -p work_dirs/transfusion_kitti_smoke
python -c "import torch, mmcv, mmengine, mmdet, mmdet3d; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), mmcv.__version__, mmengine.__version__, mmdet.__version__, mmdet3d.__version__)" \
  | tee work_dirs/transfusion_kitti_smoke/environment.txt
```

Expected: H800, capability `(9, 0)`, and the fixed environment contract.

- [ ] **Step 2: Run LiDAR one-sample forward/backward**

```bash
python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --device cuda:0 --mode backward \
  | tee work_dirs/transfusion_kitti_smoke/lidar.log
```

Expected: finite losses and `BACKWARD_OK`.

- [ ] **Step 3: Run fusion one-sample forward/backward**

```bash
python projects/TransFusionKITTI/tools/smoke_test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --device cuda:0 --mode backward \
  | tee work_dirs/transfusion_kitti_smoke/fusion.log
```

Expected: finite losses, no gradients in frozen components, gradients in image-fusion modules, and `BACKWARD_OK`.

- [x] **Step 4: Run short training and validation**

```bash
python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti_smoke \
  --cfg-options train_cfg.max_epochs=1 val_cfg=None \
                val_dataloader=None val_evaluator=None \
                train_dataloader.batch_size=1
```

After stage checkpoints exist, repeat for LC with `train_cfg.max_epochs=1`. Expected: an epoch completes, checkpoints are written, and no NaN/Inf loss appears.

Verified on H800: LC loaded the merged LiDAR/image checkpoint without sparse
encoder shape mismatches, completed 128 iterations over 64 unique samples with
`RepeatDataset(times=2)`, and saved `epoch_1.pth` using FP32.

- [x] **Step 5: Run validation metric**

```bash
python tools/test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  work_dirs/transfusion_lc_kitti_smoke/epoch_1.pth
```

Expected: `KittiMetric` prints bbox, BEV and 3D AP11/AP40, including `Kitti metric/pred_instances_3d/KITTI/Overall_3D_AP40_moderate`.

Verified on H800: inference completed for all 3769 validation samples and
printed the full KITTI bbox/BEV/3D AP11 and AP40 metric set. The near-zero AP
from the 64-sample smoke checkpoint is excluded from research conclusions.

- [x] **Step 6: Mark baseline status accurately**

After these checks, documentation may state “框架迁移和 KITTI 数据流已跑通”. It must not state “多模态方法已验证有效” until complete L/LC experiments and repeated runs support that conclusion.
