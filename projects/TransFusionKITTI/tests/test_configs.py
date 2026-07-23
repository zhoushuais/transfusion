# flake8: noqa: E402
import pytest

pytest.importorskip('mmengine')

from mmengine.config import Config


def test_lidar_config_contract():
    cfg = Config.fromfile(
        'projects/TransFusionKITTI/configs/transfusion_l_kitti.py',
        import_custom_modules=False)
    assert cfg.model.type == 'TransFusionKITTIDetector'
    assert cfg.model.fuse_img is False
    assert cfg.model.bbox_head.num_classes == 3
    assert 'vel' not in cfg.model.bbox_head.common_heads
    assert cfg.model.bbox_head.bbox_coder.code_size == 8
    assert tuple(cfg.train_dataloader.dataset.dataset.metainfo['classes']) == (
        'Pedestrian',
        'Cyclist',
        'Car',
    )
    assert cfg.default_hooks.checkpoint.save_best == (
        'Kitti metric/pred_instances_3d/KITTI/'
        'Overall_3D_AP40_moderate')


def test_lc_config_uses_single_camera_without_geometry_augmentation():
    cfg = Config.fromfile(
        'projects/TransFusionKITTI/configs/transfusion_lc_kitti.py',
        import_custom_modules=False)
    assert cfg.model.fuse_img is True
    assert cfg.model.bbox_head.fuse_img is True
    assert cfg.model.bbox_head.num_views == 1
    pipeline_types = [item['type'] for item in cfg.train_pipeline]
    assert 'LoadImageFromFileMono3D' in pipeline_types
    assert 'ObjectSample' not in pipeline_types
    assert 'ObjectNoise' not in pipeline_types
    assert 'RandomFlip3D' not in pipeline_types
    assert 'GlobalRotScaleTrans' not in pipeline_types
    assert cfg.train_cfg.max_epochs == 6
    assert 'cycle_momentum' not in cfg.param_scheduler[0]


def test_2d_pretraining_config_contract():
    cfg = Config.fromfile(
        'projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py')
    assert cfg.model.type == 'FasterRCNN'
    assert cfg.model.roi_head.bbox_head.num_classes == 3
    assert cfg.train_dataloader.dataset.ann_file.endswith(
        'kitti_2d_train.json')
    assert tuple(cfg.train_dataloader.dataset.metainfo['classes']) == (
        'Pedestrian',
        'Cyclist',
        'Car',
    )
