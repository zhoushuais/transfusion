"""Matching costs used by the KITTI TransFusion head."""

import torch
from torch import Tensor

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class TransFusionKITTIBBoxBEVL1Cost:
    """L1 cost between normalized BEV box centers."""

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def __call__(self, bboxes: Tensor, gt_bboxes: Tensor,
                 train_cfg: dict) -> Tensor:
        pc_start = bboxes.new_tensor(train_cfg['point_cloud_range'][:2])
        pc_extent = bboxes.new_tensor(
            train_cfg['point_cloud_range'][3:5]) - pc_start
        pred_xy = (bboxes[:, :2] - pc_start) / pc_extent
        gt_xy = (gt_bboxes[:, :2] - pc_start) / pc_extent
        return torch.cdist(pred_xy, gt_xy, p=1) * self.weight


@TASK_UTILS.register_module()
class TransFusionKITTIIoU3DCost:
    """Negative 3D IoU cost."""

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def __call__(self, iou: Tensor) -> Tensor:
        return -iou * self.weight
