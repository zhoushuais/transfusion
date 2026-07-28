"""Diagnose a LiDAR-only TransFusion KITTI Stage-1 checkpoint."""

import argparse
import copy
import math
from pathlib import Path
import statistics
from typing import Iterable


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


def _require_finite(name, value) -> None:
    import torch

    if value.is_floating_point() and not torch.isfinite(value).all():
        raise RuntimeError(f'{name} contains NaN or Inf')


def numeric_summary(values: Iterable[float]) -> dict:
    values = [float(value) for value in values]
    if not values:
        return dict(count=0, mean=None, median=None, p90=None)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError('summary values contain NaN or Inf')
    ordered = sorted(values)
    rank = 0.9 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    p90 = ordered[lower]
    if upper != lower:
        p90 += (ordered[upper] - ordered[lower]) * (rank - lower)
    return dict(
        count=len(ordered),
        mean=float(statistics.fmean(ordered)),
        median=float(statistics.median(ordered)),
        p90=float(p90),
    )


def flat_indices_to_xy(indices, width: int):
    import torch

    if width <= 0:
        raise ValueError('feature-map width must be positive')
    return torch.stack(
        (indices.remainder(width),
         torch.div(indices, width, rounding_mode='floor')),
        dim=-1,
    ).float()


def select_queries(localized_heatmap, num_proposals: int) -> dict:
    if localized_heatmap.ndim != 4 or localized_heatmap.shape[0] != 1:
        raise ValueError('localized_heatmap must have shape [1,C,H,W]')
    _, _, height, width = localized_heatmap.shape
    spatial_size = height * width
    flat = localized_heatmap.flatten(2)
    top = flat.reshape(1, -1).argsort(
        dim=-1, descending=True)[..., :num_proposals]
    labels = top // spatial_size
    indices = top % spatial_size
    scores = flat.gather(2,
                         indices[:, None].expand(-1, flat.shape[1], -1))
    return dict(
        labels=labels,
        indices=indices,
        xy=flat_indices_to_xy(indices, width),
        class_scores=scores,
    )


def query_metrics(query_xy, query_labels, gt_xy, gt_labels) -> dict:
    import torch

    nearest = gt_xy.new_full((len(gt_xy),), float('inf'))
    no_same_class = torch.ones(
        len(gt_xy), dtype=torch.bool, device=gt_xy.device)
    for gt_index, label in enumerate(gt_labels):
        mask = query_labels == label
        if mask.any():
            no_same_class[gt_index] = False
            nearest[gt_index] = torch.linalg.vector_norm(
                query_xy[mask] - gt_xy[gt_index], dim=1).min()
    return dict(
        nearest_distance=nearest,
        recall={radius: nearest <= radius for radius in RECALL_RADII},
        no_same_class=no_same_class,
    )


def absolute_yaw_error(pred_yaw, gt_yaw):
    import torch

    delta = pred_yaw - gt_yaw
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()


def strict_iou_threshold(class_label: int) -> float:
    if class_label < 0 or class_label >= len(CLASS_NAMES):
        raise ValueError(f'invalid KITTI class label: {class_label}')
    return STRICT_IOU[CLASS_NAMES[class_label]]


def aligned_bev_iou(pred_boxes, gt_boxes):
    from mmcv.ops import box_iou_rotated

    if len(pred_boxes) == 0:
        return pred_boxes.new_zeros((0,))
    pred_bev = pred_boxes[:, [0, 1, 3, 4, 6]]
    gt_bev = gt_boxes[:, [0, 1, 3, 4, 6]]
    return box_iou_rotated(pred_bev, gt_bev, aligned=True)


def build_match_record(
    pred_boxes,
    gt_boxes,
    gt_labels,
    query_labels,
    decoder_logits,
    query_heatmap_score,
    assignment,
    bev_iou_fn=aligned_bev_iou,
) -> dict:
    query_indices = (assignment.gt_inds > 0).nonzero(as_tuple=False).flatten()
    gt_indices = assignment.gt_inds[query_indices] - 1
    matched_pred = pred_boxes[query_indices]
    matched_gt = gt_boxes[gt_indices]
    matched_labels = gt_labels[gt_indices]
    label_match = query_labels[query_indices] == matched_labels
    decoder_score = decoder_logits[0, matched_labels,
                                   query_indices].sigmoid()
    query_score = query_heatmap_score[0, matched_labels, query_indices]
    final_score = decoder_score * query_score * label_match.float()
    thresholds = matched_pred.new_tensor([
        strict_iou_threshold(int(label.item())) for label in matched_labels
    ])
    iou_3d = assignment.max_overlaps[query_indices]
    return dict(
        gt_labels=matched_labels,
        query_indices=query_indices,
        gt_indices=gt_indices,
        center_abs_error=(matched_pred[:, :3] - matched_gt[:, :3]).abs(),
        dim_abs_error=(matched_pred[:, 3:6] - matched_gt[:, 3:6]).abs(),
        yaw_abs_error=absolute_yaw_error(matched_pred[:, 6],
                                         matched_gt[:, 6]),
        bev_iou=bev_iou_fn(matched_pred, matched_gt),
        iou_3d=iou_3d,
        strict_iou_pass=iou_3d >= thresholds,
        query_label_match=label_match,
        decoder_gt_score=decoder_score,
        final_gt_score=final_score,
    )
