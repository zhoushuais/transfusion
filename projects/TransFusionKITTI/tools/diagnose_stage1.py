"""Diagnose a LiDAR-only TransFusion KITTI Stage-1 checkpoint."""

import argparse
import copy
import json
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
    if tuple(cfg.get('class_names', ())) != CLASS_NAMES:
        raise ValueError(
            'Stage 1 diagnostics require KITTI class order: '
            'Pedestrian, Cyclist, Car')
    if bbox_head is None or int(bbox_head.get('num_classes', -1)) != len(
            CLASS_NAMES):
        raise ValueError('Stage 1 diagnostics require exactly 3 KITTI classes')
    if int(bbox_head.get('num_decoder_layers', -1)) != 1:
        raise ValueError('Stage 1 diagnostics require one decoder layer')


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
        xy=flat_indices_to_xy(indices, width) + 0.5,
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


def unwrap_prediction(raw_output) -> dict:
    if (not isinstance(raw_output, tuple) or len(raw_output) != 1 or
            not isinstance(raw_output[0], list) or len(raw_output[0]) != 1):
        raise RuntimeError('expected a single-level TransFusion prediction')
    return raw_output[0][0]


def validate_feature_map_shape(actual_shape, train_cfg) -> None:
    expected_shape = (
        int(train_cfg['grid_size'][1] // train_cfg['out_size_factor']),
        int(train_cfg['grid_size'][0] // train_cfg['out_size_factor']),
    )
    if tuple(actual_shape) != expected_shape:
        raise RuntimeError(
            f'feature map shape {tuple(actual_shape)} does not match '
            f'config-derived shape {expected_shape}')


def verify_query_reconstruction(selected: dict, prediction: dict) -> None:
    import torch

    if not torch.equal(selected['labels'], prediction['query_labels']):
        raise RuntimeError('reconstructed query labels differ from production')
    try:
        torch.testing.assert_close(
            selected['class_scores'],
            prediction['query_heatmap_score'],
            atol=1e-6,
            rtol=1e-5,
        )
    except AssertionError as error:
        raise RuntimeError(
            'reconstructed query scores differ from production') from error


def filter_target_ground_truth(boxes, labels):
    valid = (labels >= 0) & (labels < len(CLASS_NAMES))
    ignored = int((~valid).sum().item())
    return boxes[valid], labels[valid], ignored


def gt_centers_to_cells(gt_boxes, train_cfg):
    centers = gt_boxes[:, :2]
    pc_range = centers.new_tensor(train_cfg['point_cloud_range'][:2])
    voxel_size = centers.new_tensor(train_cfg['voxel_size'][:2])
    return (centers - pc_range) / voxel_size / train_cfg['out_size_factor']


def diagnose_batch(model, batch: dict) -> dict:
    import torch
    from mmengine.structures import InstanceData

    inputs = batch['inputs']
    data_samples = batch['data_samples']
    if len(data_samples) != 1:
        raise RuntimeError('diagnostics require batch_size=1')
    sample = data_samples[0]
    metas = [sample.metainfo]

    raw_output = model(inputs, data_samples, mode='tensor')
    prediction = unwrap_prediction(raw_output)
    required = {
        'dense_heatmap',
        'heatmap',
        'center',
        'height',
        'dim',
        'rot',
        'query_labels',
        'query_heatmap_score',
    }
    missing = required.difference(prediction)
    if missing:
        raise RuntimeError(f'prediction is missing keys: {sorted(missing)}')
    for name in required:
        _require_finite(name, prediction[name])

    prediction_device = prediction['center'].device
    raw_gt = sample.gt_instances_3d
    gt_boxes = raw_gt.bboxes_3d.tensor.to(prediction_device)
    gt_labels = raw_gt.labels_3d.to(prediction_device)
    gt_boxes, gt_labels, ignored_gt_count = filter_target_ground_truth(
        gt_boxes, gt_labels)

    head = model.bbox_head
    dense_probability = prediction['dense_heatmap'].sigmoid()
    validate_feature_map_shape(dense_probability.shape[-2:], head.train_cfg)
    localized = head._local_max_heatmap(dense_probability)
    selected = select_queries(localized, head.num_proposals)
    verify_query_reconstruction(selected, prediction)

    gt_xy = gt_centers_to_cells(gt_boxes, head.train_cfg)
    height, width = dense_probability.shape[-2:]
    if ((gt_xy[:, 0] < 0).any() or (gt_xy[:, 0] >= width).any() or
            (gt_xy[:, 1] < 0).any() or (gt_xy[:, 1] >= height).any()):
        raise RuntimeError('filtered GT center is outside the feature map')
    gt_int = gt_xy.to(torch.long)
    dense_gt_score = dense_probability[
        0, gt_labels, gt_int[:, 1], gt_int[:, 0]]
    query_record = query_metrics(
        selected['xy'][0], selected['labels'][0], gt_xy, gt_labels)

    num_proposals = head.num_proposals
    decoder_logits = prediction['heatmap'][..., -num_proposals:]
    decoded = head.bbox_coder.decode(
        decoder_logits.detach(),
        prediction['rot'][..., -num_proposals:].detach(),
        prediction['dim'][..., -num_proposals:].detach(),
        prediction['center'][..., -num_proposals:].detach(),
        prediction['height'][..., -num_proposals:].detach(),
        filter=False,
    )[0]['bboxes']
    _require_finite('decoded boxes', decoded)
    pred_instances = InstanceData(
        bboxes=decoded,
        scores=decoder_logits[0].transpose(0, 1),
    )
    gt_instances = InstanceData(bboxes=gt_boxes, labels=gt_labels)
    assignment = head.bbox_assigner.assign(
        pred_instances, gt_instances, head.train_cfg)
    match_record = build_match_record(
        decoded,
        gt_boxes,
        gt_labels,
        prediction['query_labels'][0],
        decoder_logits,
        prediction['query_heatmap_score'],
        assignment,
    )
    final_prediction = head.predict_by_feat(raw_output, metas)[0]
    return dict(
        gt_labels=gt_labels.detach().cpu(),
        ignored_gt_count=ignored_gt_count,
        dense_gt_score=dense_gt_score.detach().cpu(),
        nearest_query_distance=query_record['nearest_distance'].detach().cpu(),
        query_recall={
            radius: value.detach().cpu()
            for radius, value in query_record['recall'].items()
        },
        no_same_class_query=query_record['no_same_class'].detach().cpu(),
        query_labels=selected['labels'][0].detach().cpu(),
        match={
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else
            value
            for key, value in match_record.items()
        },
        prediction_labels=final_prediction.labels_3d.detach().cpu(),
        prediction_scores=final_prediction.scores_3d.detach().cpu(),
        dense_shape=tuple(dense_probability.shape[-2:]),
    )


def _python_value(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def _values(records: list, key: str, nested: str = None) -> list:
    combined = []
    for record in records:
        value = record[nested][key] if nested else record[key]
        value = _python_value(value)
        if isinstance(value, (list, tuple)):
            combined.extend(value)
        else:
            combined.append(value)
    return combined


def grouped_summary(values, labels, class_names=CLASS_NAMES) -> dict:
    values = list(values)
    labels = [int(label) for label in labels]
    if len(values) != len(labels):
        raise ValueError('values and labels must have equal length')
    return dict(
        overall=numeric_summary(values),
        by_class={
            name: numeric_summary(
                value for value, label in zip(values, labels)
                if label == class_index)
            for class_index, name in enumerate(class_names)
        },
    )


def _counts(labels) -> dict:
    labels = [int(label) for label in labels]
    return dict(
        overall=len(labels),
        by_class={
            name: sum(label == index for label in labels)
            for index, name in enumerate(CLASS_NAMES)
        },
    )


def aggregate_records(records: list) -> dict:
    if not records:
        raise ValueError('at least one diagnostic record is required')
    gt_labels = _values(records, 'gt_labels')
    match_labels = _values(records, 'gt_labels', nested='match')
    prediction_labels = _values(records, 'prediction_labels')
    nearest = _values(records, 'nearest_query_distance')
    finite_nearest = [
        (value, label) for value, label in zip(nearest, gt_labels)
        if math.isfinite(float(value))
    ]

    decoder_matches = {}
    for metric, axes in (
        ('center_abs_error', ('x', 'y', 'z')),
        ('dim_abs_error', ('dx', 'dy', 'dz')),
    ):
        rows = _values(records, metric, nested='match')
        decoder_matches[metric] = {
            axis: grouped_summary(
                [row[axis_index] for row in rows], match_labels)
            for axis_index, axis in enumerate(axes)
        }
    for metric in (
        'yaw_abs_error',
        'bev_iou',
        'iou_3d',
        'strict_iou_pass',
        'query_label_match',
    ):
        decoder_matches[metric] = grouped_summary(
            _values(records, metric, nested='match'), match_labels)

    return dict(
        ground_truth=dict(
            count=_counts(gt_labels),
            ignored_non_target_count=sum(
                int(record['ignored_gt_count']) for record in records),
        ),
        dense_queries=dict(
            gt_center_score=grouped_summary(
                _values(records, 'dense_gt_score'), gt_labels),
            nearest_same_class_distance_cells=grouped_summary(
                [item[0] for item in finite_nearest],
                [item[1] for item in finite_nearest],
            ),
            same_class_recall_cells={
                str(int(radius)): grouped_summary(
                    [value for record in records
                     for value in _python_value(record['query_recall'][radius])],
                    gt_labels,
                )
                for radius in RECALL_RADII
            },
            no_same_class_query=grouped_summary(
                _values(records, 'no_same_class_query'), gt_labels),
            query_class_count=_counts(_values(records, 'query_labels')),
        ),
        decoder_matches=decoder_matches,
        scores=dict(
            decoder_gt_class_score=grouped_summary(
                _values(records, 'decoder_gt_score', nested='match'),
                match_labels,
            ),
            final_gt_class_score=grouped_summary(
                _values(records, 'final_gt_score', nested='match'),
                match_labels,
            ),
            final_prediction_count=_counts(prediction_labels),
            final_prediction_score=grouped_summary(
                _values(records, 'prediction_scores'), prediction_labels),
        ),
    )


def validate_max_samples(max_samples: int) -> None:
    if max_samples <= 0:
        raise ValueError('--max-samples must be positive')


def write_report(result: dict, output: Path) -> None:
    serialized = json.dumps(result, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding='utf-8')


def print_summary(result: dict, output: Path) -> None:
    dense = result['dense_queries']
    matches = result['decoder_matches']
    scores = result['scores']
    print(
        f"STAGE1_DIAGNOSTICS_OK samples="
        f"{result['metadata']['processed_samples']}")
    print('dense_gt_score:', dense['gt_center_score'])
    print('query_recall_cells:', dense['same_class_recall_cells'])
    print('matched_iou_3d:', matches['iou_3d'])
    print('query_label_match:', matches['query_label_match'])
    print('decoder_gt_score:', scores['decoder_gt_class_score'])
    print('final_gt_score:', scores['final_gt_class_score'])
    print(f'JSON: {output}')


def run(args) -> dict:
    validate_max_samples(args.max_samples)
    validate_paths(args.config, args.checkpoint)

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint
    from mmengine.utils import import_modules_from_strings

    from mmdet3d.registry import MODELS

    cfg = Config.fromfile(str(args.config))
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    validate_stage1_config(cfg)

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = MODELS.build(cfg.model)
    model.init_weights()
    load_checkpoint(
        model,
        str(args.checkpoint),
        map_location='cpu',
        strict=True,
    )
    model.to(device).eval()
    dataloader = Runner.build_dataloader(
        build_diagnostic_dataloader_cfg(cfg))

    records = []
    skipped_empty_samples = 0
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = model.data_preprocessor(raw_batch, training=False)
            record = diagnose_batch(model, batch)
            if len(record['gt_labels']) == 0:
                skipped_empty_samples += 1
                continue
            records.append(record)
            if len(records) == args.max_samples:
                break
    if not records:
        raise RuntimeError('diagnostic subset contains no valid target GT')

    dense_shapes = {tuple(record['dense_shape']) for record in records}
    if len(dense_shapes) != 1:
        raise RuntimeError(f'inconsistent feature map shapes: {dense_shapes}')
    dense_shape = next(iter(dense_shapes))
    result = dict(
        metadata=dict(
            config=str(args.config.resolve()),
            checkpoint=str(args.checkpoint.resolve()),
            device=str(device),
            processed_samples=len(records),
            skipped_empty_samples=skipped_empty_samples,
            class_names=list(CLASS_NAMES),
            bev_feature_shape=list(dense_shape),
            voxel_size=list(cfg.voxel_size),
            out_size_factor=int(cfg.out_size_factor),
            point_cloud_range=list(cfg.point_cloud_range),
        ),
        **aggregate_records(records),
    )
    write_report(result, args.output)
    print_summary(result, args.output)
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
