from pathlib import Path
import json
import math
from types import SimpleNamespace

import pytest

pytest.importorskip('mmengine')

from mmengine.config import ConfigDict

from projects.TransFusionKITTI.tools import diagnose_stage1

try:
    import torch
except ImportError:
    torch = None


requires_torch = pytest.mark.skipif(torch is None, reason='requires PyTorch')


def test_parser_requires_checkpoint_and_output():
    parser = diagnose_stage1.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['config.py'])

    args = parser.parse_args([
        'config.py', '--checkpoint', 'epoch_5.pth', '--output', 'diag.json'
    ])

    assert args.device == 'cuda:0'
    assert args.max_samples == 64


def test_validate_stage1_rejects_fusion_config():
    cfg = ConfigDict(model=ConfigDict(fuse_img=True))

    with pytest.raises(ValueError, match='fuse_img=False'):
        diagnose_stage1.validate_stage1_config(cfg)


def test_validate_stage1_locks_class_order_and_single_decoder():
    cfg = ConfigDict(
        class_names=('Pedestrian', 'Cyclist', 'Car'),
        model=ConfigDict(
            fuse_img=False,
            bbox_head=ConfigDict(num_classes=3, num_decoder_layers=1),
        ),
    )
    diagnose_stage1.validate_stage1_config(cfg)

    cfg.class_names = ('Car', 'Pedestrian', 'Cyclist')
    with pytest.raises(ValueError, match='class order'):
        diagnose_stage1.validate_stage1_config(cfg)
    cfg.class_names = diagnose_stage1.CLASS_NAMES
    cfg.model.bbox_head.num_decoder_layers = 2
    with pytest.raises(ValueError, match='decoder layer'):
        diagnose_stage1.validate_stage1_config(cfg)


def test_build_diagnostic_dataloader_cfg_is_ordered_and_unaugmented():
    cfg = ConfigDict(
        backend_args=None,
        point_cloud_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
        val_dataloader=ConfigDict(
            batch_size=1,
            num_workers=1,
            persistent_workers=True,
            sampler=ConfigDict(type='DefaultSampler', shuffle=False),
            dataset=ConfigDict(
                type='KittiDataset',
                test_mode=True,
                pipeline=[ConfigDict(type='Pack3DDetInputs')],
            ),
        ),
    )

    result = diagnose_stage1.build_diagnostic_dataloader_cfg(cfg)

    assert result.batch_size == 1
    assert result.num_workers == 0
    assert result.persistent_workers is False
    assert result.sampler.shuffle is False
    assert result.dataset.test_mode is False
    assert result.dataset.filter_empty_gt is False
    assert [item.type for item in result.dataset.pipeline] == [
        'LoadPointsFromFile',
        'LoadAnnotations3D',
        'PointsRangeFilter',
        'ObjectRangeFilter',
        'Pack3DDetInputs',
    ]


def test_validate_paths_rejects_missing_checkpoint(tmp_path: Path):
    config = tmp_path / 'config.py'
    config.write_text('model = dict()', encoding='utf-8')

    with pytest.raises(FileNotFoundError, match='checkpoint'):
        diagnose_stage1.validate_paths(config, tmp_path / 'missing.pth')


def test_numeric_summary_empty_and_finite_values_are_json_safe():
    assert diagnose_stage1.numeric_summary([]) == {
        'count': 0,
        'mean': None,
        'median': None,
        'p90': None,
    }
    result = diagnose_stage1.numeric_summary([1.0, 2.0, 3.0])
    assert result['count'] == 3
    assert result['mean'] == pytest.approx(2.0)
    assert result['median'] == pytest.approx(2.0)
    assert result['p90'] == pytest.approx(2.8)


def test_numeric_summary_rejects_non_finite_values():
    with pytest.raises(RuntimeError, match='NaN or Inf'):
        diagnose_stage1.numeric_summary([1.0, float('inf')])


def test_strict_iou_threshold_uses_kitti_class_contract():
    assert diagnose_stage1.strict_iou_threshold(0) == pytest.approx(0.5)
    assert diagnose_stage1.strict_iou_threshold(1) == pytest.approx(0.5)
    assert diagnose_stage1.strict_iou_threshold(2) == pytest.approx(0.7)
    with pytest.raises(ValueError, match='class label'):
        diagnose_stage1.strict_iou_threshold(3)


def test_unwrap_prediction_requires_expected_single_level_shape():
    prediction = {'heatmap': object()}

    assert diagnose_stage1.unwrap_prediction(([prediction],)) is prediction
    with pytest.raises(RuntimeError, match='single-level'):
        diagnose_stage1.unwrap_prediction(([],))


def test_validate_feature_map_shape_rejects_config_drift():
    train_cfg = dict(grid_size=[1408, 1600, 40], out_size_factor=8)

    diagnose_stage1.validate_feature_map_shape((200, 176), train_cfg)
    with pytest.raises(RuntimeError, match='feature map shape'):
        diagnose_stage1.validate_feature_map_shape((176, 200), train_cfg)


def test_grouped_summary_keeps_empty_class_json_safe():
    result = diagnose_stage1.grouped_summary(
        values=[0.2, 0.8],
        labels=[0, 2],
    )

    assert result['overall']['count'] == 2
    assert result['by_class']['Cyclist']['count'] == 0
    assert result['by_class']['Cyclist']['mean'] is None
    json.dumps(result, allow_nan=False)


def test_aggregate_records_has_required_top_level_sections():
    record = dict(
        gt_labels=[0],
        ignored_gt_count=2,
        dense_gt_score=[0.4],
        nearest_query_distance=[1.0],
        query_recall={radius: [radius >= 1.0]
                      for radius in diagnose_stage1.RECALL_RADII},
        no_same_class_query=[False],
        query_labels=[0, 2],
        match=dict(
            gt_labels=[0],
            center_abs_error=[[1., 2., 3.]],
            dim_abs_error=[[0.1, 0.2, 0.3]],
            yaw_abs_error=[0.1],
            bev_iou=[0.4],
            iou_3d=[0.3],
            strict_iou_pass=[False],
            query_label_match=[True],
            decoder_gt_score=[0.5],
            final_gt_score=[0.25],
        ),
        prediction_labels=[0, 2],
        prediction_scores=[0.4, 0.1],
        dense_shape=(200, 176),
    )

    result = diagnose_stage1.aggregate_records([record])

    assert set(result) == {
        'ground_truth', 'dense_queries', 'decoder_matches', 'scores'
    }
    assert result['ground_truth']['count']['overall'] == 1
    assert result['ground_truth']['ignored_non_target_count'] == 2
    json.dumps(result, allow_nan=False)


def test_aggregate_records_excludes_infinite_nearest_distance():
    record = dict(
        gt_labels=[1],
        ignored_gt_count=0,
        dense_gt_score=[0.2],
        nearest_query_distance=[float('inf')],
        query_recall={radius: [False]
                      for radius in diagnose_stage1.RECALL_RADII},
        no_same_class_query=[True],
        query_labels=[0],
        match=dict(
            gt_labels=[1],
            center_abs_error=[[1., 1., 1.]],
            dim_abs_error=[[1., 1., 1.]],
            yaw_abs_error=[1.],
            bev_iou=[0.],
            iou_3d=[0.],
            strict_iou_pass=[False],
            query_label_match=[False],
            decoder_gt_score=[0.1],
            final_gt_score=[0.],
        ),
        prediction_labels=[],
        prediction_scores=[],
        dense_shape=(200, 176),
    )

    result = diagnose_stage1.aggregate_records([record])

    nearest = result['dense_queries']['nearest_same_class_distance_cells']
    assert nearest['overall']['count'] == 0
    assert result['dense_queries']['no_same_class_query']['overall'][
        'mean'] == pytest.approx(1.0)
    json.dumps(result, allow_nan=False)


def test_validate_max_samples_requires_positive_value():
    diagnose_stage1.validate_max_samples(1)
    with pytest.raises(ValueError, match='positive'):
        diagnose_stage1.validate_max_samples(0)


def test_write_report_creates_parent_and_rejects_nan(tmp_path: Path):
    output = tmp_path / 'nested' / 'report.json'

    diagnose_stage1.write_report({'value': 1.5}, output)

    assert json.loads(output.read_text(encoding='utf-8')) == {'value': 1.5}
    with pytest.raises(ValueError):
        diagnose_stage1.write_report({'value': float('nan')}, output)


@requires_torch
def test_flat_indices_restore_xy_on_non_square_map():
    indices = torch.tensor([0, 4, 5, 13])

    result = diagnose_stage1.flat_indices_to_xy(indices, width=5)

    torch.testing.assert_close(
        result,
        torch.tensor([[0., 0.], [4., 0.], [0., 1.], [3., 2.]]),
    )


@requires_torch
def test_select_queries_preserves_class_and_xy():
    localized = torch.zeros(1, 2, 3, 5)
    localized[0, 0, 1, 4] = 0.9
    localized[0, 1, 2, 1] = 0.8

    selected = diagnose_stage1.select_queries(localized, num_proposals=2)

    assert selected['labels'][0].tolist() == [0, 1]
    torch.testing.assert_close(
        selected['xy'][0], torch.tensor([[4.5, 1.5], [1.5, 2.5]]))


@requires_torch
def test_query_metrics_use_same_class_distance():
    query_xy = torch.tensor([[4., 1.], [1., 2.], [0., 0.]])
    query_labels = torch.tensor([0, 1, 0])
    gt_xy = torch.tensor([[3., 1.], [4., 2.], [1., 2.]])
    gt_labels = torch.tensor([0, 0, 1])

    result = diagnose_stage1.query_metrics(query_xy, query_labels, gt_xy,
                                            gt_labels)

    torch.testing.assert_close(
        result['nearest_distance'], torch.tensor([1., 1., 0.]))
    assert result['recall'][1.0].tolist() == [True, True, True]
    assert result['no_same_class'].tolist() == [False, False, False]


@requires_torch
def test_wrap_angle_returns_small_absolute_boundary_error():
    error = diagnose_stage1.absolute_yaw_error(
        torch.tensor([math.pi - 0.1]),
        torch.tensor([-math.pi + 0.1]),
    )

    torch.testing.assert_close(
        error, torch.tensor([0.2]), atol=1e-6, rtol=1e-6)


@requires_torch
def test_build_match_record_uses_assigned_query_gt_pairs():
    pred_boxes = torch.tensor([
        [10., 1., -1., 4., 2., 1.5, 0.2],
        [20., 2., -1., 1., 1., 1.7, -0.2],
    ])
    gt_boxes = torch.tensor([
        [10.5, 1.5, -1.2, 3.5, 1.5, 1.4, 0.1],
        [20.5, 2.5, -0.8, 1.2, 0.8, 1.6, -0.1],
    ])
    gt_labels = torch.tensor([2, 0])
    query_labels = torch.tensor([2, 1])
    logits = torch.tensor([[[-8., 2.], [-8., 3.], [4., -8.]]])
    query_scores = torch.full((1, 3, 2), 0.5)
    assignment = SimpleNamespace(
        gt_inds=torch.tensor([1, 2]),
        max_overlaps=torch.tensor([0.8, 0.4]),
    )

    record = diagnose_stage1.build_match_record(
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        query_labels=query_labels,
        decoder_logits=logits,
        query_heatmap_score=query_scores,
        assignment=assignment,
        bev_iou_fn=lambda pred, gt: torch.tensor([0.7, 0.3]),
    )

    assert record['gt_labels'].tolist() == [2, 0]
    torch.testing.assert_close(record['center_abs_error'][0],
                               torch.tensor([0.5, 0.5, 0.2]))
    torch.testing.assert_close(record['dim_abs_error'][1],
                               torch.tensor([0.2, 0.2, 0.1]))
    torch.testing.assert_close(record['iou_3d'], torch.tensor([0.8, 0.4]))
    assert record['query_label_match'].tolist() == [True, False]
    assert record['strict_iou_pass'].tolist() == [True, False]
    assert record['final_gt_score'][1].item() == 0.0


@requires_torch
def test_verify_query_reconstruction_rejects_coordinate_drift():
    selected = dict(
        labels=torch.tensor([[0, 1]]),
        class_scores=torch.tensor([[[0.8, 0.2], [0.1, 0.7]]]),
    )
    prediction = dict(
        query_labels=torch.tensor([[1, 0]]),
        query_heatmap_score=selected['class_scores'],
    )

    with pytest.raises(RuntimeError, match='query labels'):
        diagnose_stage1.verify_query_reconstruction(selected, prediction)


@requires_torch
def test_filter_target_ground_truth_removes_negative_labels():
    boxes = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    labels = torch.tensor([0, -1, 2])

    filtered_boxes, filtered_labels, ignored = (
        diagnose_stage1.filter_target_ground_truth(boxes, labels))

    assert filtered_labels.tolist() == [0, 2]
    torch.testing.assert_close(filtered_boxes, boxes[[0, 2]])
    assert ignored == 1
