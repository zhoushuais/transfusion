"""KITTI-specific data validation transforms."""

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ValidateKittiCalibration(BaseTransform):
    """Validate and normalize single-camera KITTI calibration matrices."""

    def transform(self, results: dict) -> dict:
        sample_id = results.get('sample_idx', '<unknown>')
        if 'lidar2img' not in results:
            raise ValueError(f'sample {sample_id}: missing lidar2img')
        matrix = np.asarray(results['lidar2img'])
        if matrix.ndim == 3 and matrix.shape[0] == 1:
            matrix = matrix[0]
        if matrix.shape not in ((3, 4), (4, 4)):
            raise ValueError(
                f'sample {sample_id}: lidar2img must be 3x4 or 4x4, '
                f'got {matrix.shape}')
        if not np.isfinite(matrix).all():
            raise ValueError(
                f'sample {sample_id}: lidar2img contains non-finite values')
        if matrix.shape == (3, 4):
            normalized = np.eye(4, dtype=matrix.dtype)
            normalized[:3] = matrix
            matrix = normalized
        results['lidar2img'] = matrix

        homography = np.asarray(
            results.get('homography_matrix', np.eye(3, dtype=np.float32)))
        if homography.shape != (3, 3) or not np.isfinite(homography).all():
            raise ValueError(
                f'sample {sample_id}: homography_matrix must be finite 3x3')
        results['homography_matrix'] = homography
        return results
