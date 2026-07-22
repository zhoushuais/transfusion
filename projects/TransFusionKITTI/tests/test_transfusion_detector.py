# flake8: noqa: E402
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv')
pytest.importorskip('mmengine')
pytest.importorskip('mmdet')
from torch import nn

from projects.TransFusionKITTI.transfusion_kitti.models import \
    TransFusionKITTIDetector


def test_freeze_module_disables_grad_and_bn_updates():
    module = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    TransFusionKITTIDetector.freeze_module(module)
    module.train()
    TransFusionKITTIDetector.freeze_module(module)
    assert not any(parameter.requires_grad
                   for parameter in module.parameters())
    assert not module[1].training


def test_extract_img_feat_selects_p2_and_lidar_mode_ignores_images():
    detector = object.__new__(TransFusionKITTIDetector)
    nn.Module.__init__(detector)
    detector.fuse_img = True
    detector.img_feature_level = 0
    detector.img_backbone = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1))
    detector.img_neck = None
    feature = detector.extract_img_feat(torch.randn(2, 3, 16, 16))
    assert feature.shape == (2, 8, 16, 16)

    detector.fuse_img = False
    assert detector.extract_img_feat(None) is None
