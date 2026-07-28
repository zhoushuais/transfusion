"""Diagnose a LiDAR-only TransFusion KITTI Stage-1 checkpoint."""

import argparse
import copy
from pathlib import Path


CLASS_NAMES = ('Pedestrian', 'Cyclist', 'Car')
RECALL_RADII = (1.0, 2.0, 4.0, 8.0)
STRICT_IOU = {'Pedestrian': 0.5, 'Cyclist': 0.5, 'Car': 0.7}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def validate_paths(config: Path, checkpoint: Path) -> None:
    if not config.is_file():
        raise FileNotFoundError(f'config does not exist: {config}')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint does not exist: {checkpoint}')


def validate_stage1_config(cfg) -> None:
    if bool(cfg.model.get('fuse_img', False)):
        raise ValueError(
            'Stage 1 diagnostics require model.fuse_img=False')
    bbox_head = cfg.model.get('bbox_head')
    if bbox_head is not None and int(bbox_head.get('num_classes', -1)) != len(
            CLASS_NAMES):
        raise ValueError('Stage 1 diagnostics require exactly 3 KITTI classes')


def build_diagnostic_dataloader_cfg(cfg):
    dataloader_cfg = copy.deepcopy(cfg.val_dataloader)
    dataloader_cfg.batch_size = 1
    dataloader_cfg.num_workers = 0
    dataloader_cfg.persistent_workers = False
    dataloader_cfg.sampler.shuffle = False

    dataset_cfg = dataloader_cfg.dataset
    dataset_cfg.test_mode = False
    dataset_cfg.filter_empty_gt = False
    backend_args = cfg.get('backend_args', None)
    point_cloud_range = list(cfg.point_cloud_range)
    dataset_cfg.pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            backend_args=backend_args,
        ),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
        ),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(
            type='Pack3DDetInputs',
            keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        ),
    ]
    return dataloader_cfg
