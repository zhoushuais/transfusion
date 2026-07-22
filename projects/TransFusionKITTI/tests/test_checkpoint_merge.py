# flake8: noqa: E402
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from projects.TransFusionKITTI.tools.merge_pretrained_weights import merge


def test_merge_maps_image_backbone_and_neck(tmp_path: Path):
    lidar = tmp_path / 'lidar.pth'
    image = tmp_path / 'image.pth'
    output = tmp_path / 'merged.pth'
    torch.save({'state_dict': {'backbone.weight': torch.ones(1)}}, lidar)
    torch.save(
        {
            'state_dict': {
                'backbone.conv.weight': torch.full((1, ), 2.0),
                'neck.fpn.weight': torch.full((1, ), 3.0),
                'roi_head.weight': torch.full((1, ), 4.0),
            }
        },
        image,
    )
    merged, report = merge(lidar, image, output)
    assert merged['state_dict']['img_backbone.conv.weight'].item() == 2
    assert merged['state_dict']['img_neck.fpn.weight'].item() == 3
    assert 'img_roi_head.weight' not in merged['state_dict']
    assert report['mapped_image_keys'] == 2
    assert output.is_file()


def test_merge_rejects_zero_image_coverage(tmp_path: Path):
    lidar = tmp_path / 'lidar.pth'
    image = tmp_path / 'image.pth'
    output = tmp_path / 'merged.pth'
    torch.save({'state_dict': {'backbone.weight': torch.ones(1)}}, lidar)
    torch.save({'state_dict': {'roi_head.weight': torch.ones(1)}}, image)
    with pytest.raises(RuntimeError, match='no image backbone or neck keys'):
        merge(lidar, image, output)


def test_merge_rejects_shape_collision(tmp_path: Path):
    lidar = tmp_path / 'lidar.pth'
    image = tmp_path / 'image.pth'
    output = tmp_path / 'merged.pth'
    torch.save({'state_dict': {
        'img_backbone.conv.weight': torch.ones(2)
    }}, lidar)
    torch.save({'state_dict': {'backbone.conv.weight': torch.ones(3)}}, image)
    with pytest.raises(RuntimeError, match='shape collision'):
        merge(lidar, image, output)
