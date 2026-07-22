"""Velocity-free bounding-box coder for KITTI TransFusion."""

from typing import Dict, List, Optional, Sequence

import torch
from mmdet.models.task_modules import BaseBBoxCoder
from torch import Tensor

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class TransFusionKITTIBBoxCoder(BaseBBoxCoder):
    """Encode and decode seven-parameter KITTI LiDAR boxes.

    KITTI has no velocity annotations, so the encoded representation is fixed
    to ``(x, y, z, log(dx), log(dy), log(dz), sin(yaw), cos(yaw))``.
    """

    def __init__(
        self,
        pc_range: Sequence[float],
        out_size_factor: int,
        voxel_size: Sequence[float],
        post_center_range: Optional[Sequence[float]] = None,
        score_threshold: Optional[float] = None,
        code_size: int = 8,
    ) -> None:
        if code_size != 8:
            raise ValueError('KITTI TransFusion bbox code_size must be 8')
        self.pc_range = tuple(pc_range)
        self.out_size_factor = out_size_factor
        self.voxel_size = tuple(voxel_size)
        self.post_center_range = (None if post_center_range is None else
                                  tuple(post_center_range))
        self.score_threshold = score_threshold
        self.code_size = code_size

    def encode(self, dst_boxes: Tensor) -> Tensor:
        """Encode bottom-centered LiDAR boxes without velocity."""
        if dst_boxes.shape[-1] < 7:
            raise ValueError('dst_boxes must contain at least seven values')
        targets = dst_boxes.new_zeros((dst_boxes.shape[0], self.code_size))
        targets[:, 0] = (dst_boxes[:, 0] - self.pc_range[0]) / (
            self.out_size_factor * self.voxel_size[0])
        targets[:, 1] = (dst_boxes[:, 1] - self.pc_range[1]) / (
            self.out_size_factor * self.voxel_size[1])
        targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5
        targets[:, 3:6] = dst_boxes[:, 3:6].log()
        targets[:, 6] = torch.sin(dst_boxes[:, 6])
        targets[:, 7] = torch.cos(dst_boxes[:, 6])
        return targets

    def decode(
        self,
        heatmap: Tensor,
        rot: Tensor,
        dim: Tensor,
        center: Tensor,
        height: Tensor,
        vel: Optional[Tensor] = None,
        filter: bool = False,
    ) -> List[Dict[str, Tensor]]:
        """Decode predictions while leaving all caller-owned tensors intact."""
        if vel is not None:
            raise ValueError('KITTI TransFusion does not decode velocity')

        final_scores, final_preds = heatmap.max(dim=1)
        center = center.clone()
        dim = dim.exp()
        center[:, 0] = (
            center[:, 0] * self.out_size_factor * self.voxel_size[0] +
            self.pc_range[0])
        center[:, 1] = (
            center[:, 1] * self.out_size_factor * self.voxel_size[1] +
            self.pc_range[1])
        height = height - dim[:, 2:3] * 0.5
        yaw = torch.atan2(rot[:, 0:1], rot[:, 1:2])
        boxes = torch.cat([center, height, dim, yaw], dim=1).permute(0, 2, 1)

        masks = torch.ones_like(final_scores, dtype=torch.bool)
        if filter:
            if self.post_center_range is None:
                raise ValueError(
                    'post_center_range is required when filter=True')
            post_range = boxes.new_tensor(self.post_center_range)
            masks &= (boxes[..., :3] >= post_range[:3]).all(dim=-1)
            masks &= (boxes[..., :3] <= post_range[3:]).all(dim=-1)
            if self.score_threshold is not None:
                masks &= final_scores > self.score_threshold

        predictions = []
        for batch_index in range(heatmap.shape[0]):
            mask = masks[batch_index]
            predictions.append(
                dict(
                    bboxes=boxes[batch_index, mask],
                    scores=final_scores[batch_index, mask],
                    labels=final_preds[batch_index, mask],
                ))
        return predictions
