# flake8: noqa: E402
import numpy as np
import pytest

pytest.importorskip('mmcv')
pytest.importorskip('mmengine')
pytest.importorskip('mmdet')

from projects.TransFusionKITTI.transfusion_kitti.datasets import \
    ValidateKittiCalibration


def test_calibration_validator_reports_sample_id():
    transform = ValidateKittiCalibration()
    with pytest.raises(ValueError, match='sample 42.*lidar2img'):
        transform(dict(sample_idx=42, lidar2img=np.eye(3)))


def test_calibration_validator_normalizes_3x4_and_adds_homography():
    result = ValidateKittiCalibration()(
        dict(sample_idx=7, lidar2img=np.eye(3, 4)))
    assert result['lidar2img'].shape == (4, 4)
    np.testing.assert_array_equal(result['homography_matrix'], np.eye(3))
