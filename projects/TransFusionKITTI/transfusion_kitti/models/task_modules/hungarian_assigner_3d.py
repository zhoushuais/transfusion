"""Hungarian assignment for modern TransFusion prediction structures."""

from typing import Optional

import torch
from mmdet.models.task_modules import AssignResult, BaseAssigner
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import TASK_UTILS

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def _box_tensor(boxes) -> Tensor:
    """Return the tensor representation of a box container."""
    return boxes.tensor if hasattr(boxes, 'tensor') else boxes


@TASK_UTILS.register_module()
class TransFusionKITTIHungarianAssigner3D(BaseAssigner):
    """Assign TransFusion queries with classification, BEV and IoU costs."""

    def __init__(
        self,
        cls_cost: dict,
        reg_cost: dict,
        iou_cost: dict,
        iou_calculator: dict,
    ) -> None:
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_cost = TASK_UTILS.build(iou_cost)
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        train_cfg: dict,
        gt_instances_ignore: Optional[InstanceData] = None,
    ) -> AssignResult:
        """Assign predictions to ground truths using a global minimum cost."""
        if gt_instances_ignore is not None and len(gt_instances_ignore) > 0:
            raise NotImplementedError('ignored GT boxes are not supported')

        pred_boxes = _box_tensor(pred_instances.bboxes)
        gt_boxes = _box_tensor(gt_instances.bboxes)
        num_preds = pred_boxes.shape[0]
        num_gts = gt_boxes.shape[0]
        assigned_gt_inds = pred_boxes.new_full((num_preds, ),
                                               -1,
                                               dtype=torch.long)
        assigned_labels = pred_boxes.new_full((num_preds, ),
                                              -1,
                                              dtype=torch.long)
        max_overlaps = pred_boxes.new_zeros(num_preds)

        if num_gts == 0 or num_preds == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels,
            )

        cls_cost = self.cls_cost(pred_instances, gt_instances)
        reg_cost = self.reg_cost(pred_boxes, gt_boxes, train_cfg)
        iou = self.iou_calculator(pred_boxes, gt_boxes)
        total_cost = cls_cost + reg_cost + self.iou_cost(iou)

        if linear_sum_assignment is None:
            raise ImportError('scipy is required for Hungarian assignment')
        row_indices, col_indices = linear_sum_assignment(
            total_cost.detach().cpu().numpy())
        row_indices = torch.as_tensor(row_indices, device=pred_boxes.device)
        col_indices = torch.as_tensor(col_indices, device=pred_boxes.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[row_indices] = col_indices + 1
        assigned_labels[row_indices] = gt_instances.labels[col_indices]
        max_overlaps[row_indices] = iou[row_indices, col_indices]
        return AssignResult(
            num_gts,
            assigned_gt_inds,
            max_overlaps,
            labels=assigned_labels,
        )
