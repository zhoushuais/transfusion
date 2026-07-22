# flake8: noqa: E402
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv')
pytest.importorskip('mmdet')

from projects.TransFusionKITTI.transfusion_kitti.models.projection import \
    project_lidar_points


def test_projection_rejects_behind_camera_and_applies_homography():
    points = torch.tensor([[2.0, 1.0, 10.0], [2.0, 1.0, -1.0]])
    lidar2img = torch.tensor([
        [100.0, 0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    homography = torch.tensor([[2.0, 0.0, 5.0], [0.0, 2.0, 7.0],
                               [0.0, 0.0, 1.0]])
    pixels, valid = project_lidar_points(
        points, lidar2img, homography, image_shape=(100, 100))
    torch.testing.assert_close(pixels[0], torch.tensor([45.0, 27.0]))
    assert valid.tolist() == [True, False]


def test_projection_handles_empty_and_single_visible_point():
    matrix = torch.eye(4)
    empty_pixels, empty_valid = project_lidar_points(
        torch.zeros(0, 3), matrix, torch.eye(3), image_shape=(10, 10))
    assert empty_pixels.shape == (0, 2)
    assert empty_valid.shape == (0, )

    pixels, valid = project_lidar_points(
        torch.tensor([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]),
        matrix,
        torch.eye(3),
        image_shape=(10, 10),
    )
    assert pixels.shape == (2, 2)
    assert valid.tolist() == [True, False]
