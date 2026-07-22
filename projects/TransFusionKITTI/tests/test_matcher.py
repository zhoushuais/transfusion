# flake8: noqa: E402
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv')
pytest.importorskip('mmdet')
from mmengine.structures import InstanceData

from projects.TransFusionKITTI.transfusion_kitti.models.task_modules import \
    TransFusionKITTIHungarianAssigner3D


class DiagonalIoU:

    def __call__(self, pred_boxes, gt_boxes):
        distances = torch.cdist(pred_boxes[:, :2], gt_boxes[:, :2], p=1)
        return (1.0 - distances / 10.0).clamp(min=0.0)


def make_assigner():
    assigner = TransFusionKITTIHungarianAssigner3D(
        iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
        cls_cost=dict(
            type='mmdet.FocalLossCost', weight=0.15, alpha=0.25, gamma=2.0),
        reg_cost=dict(type='TransFusionKITTIBBoxBEVL1Cost', weight=0.25),
        iou_cost=dict(type='TransFusionKITTIIoU3DCost', weight=0.25),
    )
    assigner.iou_calculator = DiagonalIoU()
    return assigner


def test_hungarian_assigner_matches_same_class_nearest_boxes():
    pred = InstanceData(
        bboxes=torch.tensor([
            [10.0, 0.0, -1.0, 4.0, 2.0, 1.5, 0.0],
            [30.0, 0.0, -1.0, 4.0, 2.0, 1.5, 0.0],
        ]),
        scores=torch.tensor([[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]]),
    )
    gt = InstanceData(
        bboxes=torch.tensor([
            [10.1, 0.0, -1.0, 4.0, 2.0, 1.5, 0.0],
            [30.1, 0.0, -1.0, 4.0, 2.0, 1.5, 0.0],
        ]),
        labels=torch.tensor([0, 1]),
    )
    result = make_assigner().assign(
        pred,
        gt,
        dict(point_cloud_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0]),
    )
    assert result.gt_inds.tolist() == [1, 2]
    assert result.labels.tolist() == [0, 1]


def test_hungarian_assigner_marks_predictions_background_for_empty_gt():
    pred = InstanceData(bboxes=torch.zeros(2, 7), scores=torch.zeros(2, 3))
    gt = InstanceData(
        bboxes=torch.zeros(0, 7), labels=torch.zeros(0, dtype=torch.long))
    result = make_assigner().assign(
        pred,
        gt,
        dict(point_cloud_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0]),
    )
    assert result.gt_inds.tolist() == [0, 0]
    assert result.labels.tolist() == [-1, -1]
