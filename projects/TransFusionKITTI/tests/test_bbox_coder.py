# flake8: noqa: E402
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv')
pytest.importorskip('mmengine')
pytest.importorskip('mmdet')

from projects.TransFusionKITTI.transfusion_kitti.models.task_modules import \
    TransFusionKITTIBBoxCoder


def make_coder():
    return TransFusionKITTIBBoxCoder(
        pc_range=[0.0, -40.0],
        out_size_factor=8,
        voxel_size=[0.05, 0.05],
        post_center_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
        score_threshold=0.0,
        code_size=8,
    )


def test_bbox_coder_round_trip_without_velocity():
    coder = make_coder()
    boxes = torch.tensor([[12.0, -2.0, -1.5, 3.9, 1.6, 1.5, 0.25]])
    encoded = coder.encode(boxes)
    decoded = coder.decode(
        heatmap=torch.tensor([[[0.9]]]),
        rot=encoded[:, 6:8].T.unsqueeze(0),
        dim=encoded[:, 3:6].T.unsqueeze(0),
        center=encoded[:, 0:2].T.unsqueeze(0),
        height=encoded[:, 2:3].T.unsqueeze(0),
        filter=False,
    )[0]['bboxes']
    torch.testing.assert_close(decoded[0], boxes[0], atol=1e-5, rtol=1e-5)


def test_bbox_coder_decode_does_not_mutate_inputs():
    coder = make_coder()
    center = torch.tensor([[[30.0], [80.0]]])
    dim = torch.zeros(1, 3, 1)
    center_before, dim_before = center.clone(), dim.clone()
    coder.decode(
        torch.tensor([[[0.9]]]),
        torch.tensor([[[0.0], [1.0]]]),
        dim,
        center,
        torch.zeros(1, 1, 1),
        filter=False,
    )
    torch.testing.assert_close(center, center_before)
    torch.testing.assert_close(dim, dim_before)


def test_bbox_coder_filters_boxes_outside_post_center_range():
    coder = make_coder()
    decoded = coder.decode(
        heatmap=torch.tensor([[[0.9, 0.8]]]),
        rot=torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
        dim=torch.zeros(1, 3, 2),
        center=torch.tensor([[[25.0, 250.0], [100.0, 100.0]]]),
        height=torch.zeros(1, 1, 2),
        filter=True,
    )[0]
    assert len(decoded['bboxes']) == 1
