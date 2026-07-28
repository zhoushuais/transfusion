from pathlib import Path
import math

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
        selected['xy'][0], torch.tensor([[4., 1.], [1., 2.]]))


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
