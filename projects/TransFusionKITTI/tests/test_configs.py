# flake8: noqa: E402
import pytest

pytest.importorskip('mmengine')

from mmengine.config import Config


def assert_batch_protocol(cfg, micro_batch, accumulation, nominal_batch):
    assert cfg.train_dataloader.batch_size == micro_batch
    assert cfg.optim_wrapper.get('accumulative_counts') == accumulation
    assert micro_batch * accumulation == nominal_batch


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
    assert cfg.train_cfg.max_epochs == 40
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(0.0018)
    assert cfg.auto_scale_lr.enable is False
    assert [scheduler.type for scheduler in cfg.param_scheduler] == [
        'CosineAnnealingLR',
        'CosineAnnealingLR',
        'CosineAnnealingMomentum',
        'CosineAnnealingMomentum',
    ]
    assert [(scheduler.begin, scheduler.end)
            for scheduler in cfg.param_scheduler] == [
                (0, 16),
                (16, 40),
                (0, 16),
                (16, 40),
            ]
    assert [scheduler.T_max for scheduler in cfg.param_scheduler] == [
        16,
        24,
        16,
        24,
    ]
    assert [scheduler.eta_min for scheduler in
            cfg.param_scheduler] == pytest.approx([
                0.018,
                1.8e-7,
                0.85 / 0.95,
                1,
            ])
    assert all(scheduler.by_epoch is True
               for scheduler in cfg.param_scheduler)
    assert all(scheduler.convert_to_iter_based is True
               for scheduler in cfg.param_scheduler)
    assert_batch_protocol(cfg, 6, 8, 48)


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
    assert_batch_protocol(cfg, 2, 8, 16)
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(1e-4)
    assert cfg.param_scheduler[0].type == 'OneCycleLR'
    assert cfg.param_scheduler[0].total_steps == 6
    assert cfg.param_scheduler[0].eta_max == pytest.approx(1e-3)
    assert cfg.auto_scale_lr.enable is False
    assert cfg.default_hooks.checkpoint.save_best == (
        'Kitti metric/pred_instances_3d/KITTI/'
        'Overall_3D_AP40_moderate')


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
    assert cfg.train_cfg.max_epochs == 12
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(0.02)
    assert cfg.auto_scale_lr.enable is False
    assert cfg.default_hooks.checkpoint.save_best == 'coco/bbox_mAP'
    assert [scheduler.type for scheduler in cfg.param_scheduler] == [
        'LinearLR',
        'MultiStepLR',
    ]
    assert [(scheduler.begin, scheduler.end)
            for scheduler in cfg.param_scheduler] == [
                (0, 500),
                (0, 12),
            ]
    assert cfg.param_scheduler[0].start_factor == pytest.approx(0.001)
    assert cfg.param_scheduler[0].by_epoch is False
    assert cfg.param_scheduler[1].milestones == [8, 11]
    assert cfg.param_scheduler[1].gamma == pytest.approx(0.1)
    assert cfg.param_scheduler[1].by_epoch is True
    assert_batch_protocol(cfg, 4, 4, 16)
