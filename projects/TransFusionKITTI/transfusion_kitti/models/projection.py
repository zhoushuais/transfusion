"""Projection helpers for KITTI LiDAR queries and boxes."""

from typing import Sequence, Tuple

import torch
from torch import Tensor


def project_lidar_points(
    points: Tensor,
    lidar2img: Tensor,
    homography: Tensor,
    image_shape: Sequence[int],
    eps: float = 1e-5,
) -> Tuple[Tensor, Tensor]:
    """Project LiDAR points into the augmented KITTI image plane."""
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError('points must have shape [N,3]')
    if lidar2img.shape not in ((3, 4), (4, 4)):
        raise ValueError('lidar2img must have shape [3,4] or [4,4]')
    if homography.shape != (3, 3):
        raise ValueError('homography must have shape [3,3]')

    points_h = torch.cat(
        [points, points.new_ones((points.shape[0], 1))], dim=-1)
    projected = points_h @ lidar2img.transpose(0, 1)
    depth = projected[:, 2]
    depth_valid = depth > eps
    safe_depth = torch.where(depth_valid, depth, torch.ones_like(depth))
    pixels_h = torch.cat(
        [
            projected[:, :2] / safe_depth[:, None],
            projected.new_ones((points.shape[0], 1)),
        ],
        dim=-1,
    )
    pixels_aug = pixels_h @ homography.transpose(0, 1)
    aug_scale = pixels_aug[:, 2]
    aug_valid = aug_scale.abs() > eps
    safe_scale = torch.where(aug_valid, aug_scale, torch.ones_like(aug_scale))
    pixels = pixels_aug[:, :2] / safe_scale[:, None]
    height, width = image_shape[:2]
    finite = torch.isfinite(pixels).all(dim=-1)
    valid = (
        depth_valid
        & aug_valid
        & finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height))
    return pixels, valid
