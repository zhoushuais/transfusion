from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('mmengine')

from mmengine.config import ConfigDict

from projects.TransFusionKITTI.tools import probe_stage1_gradients as probe


def test_parser_requires_checkpoint_and_output():
    parser = probe.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['config.py'])
    args = parser.parse_args([
        'config.py', '--checkpoint', 'epoch_5.pth',
        '--output', 'gradient_probe.json'
    ])
    assert args.device == 'cuda:0'
    assert args.seed == 0


def test_validate_probe_config_accepts_only_fp32_stage1():
    cfg = ConfigDict(
        class_names=('Pedestrian', 'Cyclist', 'Car'),
        model=ConfigDict(
            fuse_img=False,
            bbox_head=ConfigDict(num_classes=3, num_decoder_layers=1),
        ),
        train_dataloader=ConfigDict(batch_size=6),
        optim_wrapper=ConfigDict(
            type='OptimWrapper', accumulative_counts=8,
            optimizer=ConfigDict(type='AdamW', lr=0.0018),
        ),
    )
    probe.validate_probe_config(cfg)
    cfg.optim_wrapper.type = 'AmpOptimWrapper'
    with pytest.raises(ValueError, match='FP32'):
        probe.validate_probe_config(cfg)


def test_build_probe_dataloader_cfg_preserves_batch_and_train_pipeline():
    pipeline = [ConfigDict(type='LoadPointsFromFile')]
    cfg = ConfigDict(
        train_dataloader=ConfigDict(
            batch_size=6,
            num_workers=4,
            persistent_workers=True,
            drop_last=True,
            sampler=ConfigDict(type='DefaultSampler', shuffle=True),
            dataset=ConfigDict(type='RepeatDataset', dataset=ConfigDict(
                type='KittiDataset', pipeline=pipeline)),
        ))
    result = probe.build_probe_dataloader_cfg(cfg)
    assert result.batch_size == 6
    assert result.num_workers == 0
    assert result.persistent_workers is False
    assert result.drop_last is False
    assert result.sampler.shuffle is False
    assert result.dataset.dataset.pipeline == pipeline


def test_validate_paths_requires_real_config_and_checkpoint(tmp_path: Path):
    config = tmp_path / 'config.py'
    config.write_text('model = dict()', encoding='utf-8')
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        probe.validate_paths(config, tmp_path / 'missing.pth')


try:
    import torch
except ImportError:
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason='requires PyTorch')


def test_summary_reports_values_and_rejects_nonfinite():
    assert probe._summary([1.0, 3.0]) == {
        'count': 2, 'mean': 2.0, 'min': 1.0, 'max': 3.0
    }
    with pytest.raises(RuntimeError, match='NaN or Inf'):
        probe._summary([float('nan')])


@requires_torch
def test_summarize_heatmap_counts_centers_by_class():
    target = torch.zeros(2, 3, 2, 2)
    target[0, 0, 0, 0] = 1
    target[0, 2, 1, 1] = 1
    target[1, 2, 0, 1] = 1
    logits = torch.zeros_like(target)
    result = probe.summarize_heatmap(logits, target)
    assert result['positive_centers']['by_class'] == {
        'Pedestrian': 1, 'Cyclist': 0, 'Car': 2
    }
    assert result['positive_probability']['overall']['mean'] == pytest.approx(
        0.5)
    assert result['nonfinite_target_count'] == 0


@requires_torch
def test_parameter_membership_reports_missing_parameter():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.SGD(model[0].parameters(), lr=0.1)
    result = probe.parameter_membership(dict(model.named_parameters()),
                                        optimizer)
    assert result['0.weight'] is True
    assert result['1.weight'] is False


@requires_torch
def test_gradient_and_delta_summaries_are_finite():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    gradients = probe.summarize_gradients({'weight': parameter.grad})
    before = probe.snapshot_parameters({'weight': parameter})
    with torch.no_grad():
        parameter.add_(torch.tensor([0.1, -0.2]))
    delta = probe.summarize_parameter_delta(before, {'weight': parameter})
    assert gradients['overall_norm'] == pytest.approx(5.0)
    assert gradients['nonfinite_count'] == 0
    assert delta['overall_norm'] == pytest.approx(0.2236068)
    assert delta['nonfinite_count'] == 0
    assert delta['all_zero'] is False


@requires_torch
def test_output_channel_gradient_summary_uses_three_kitti_classes():
    weight = torch.ones(3, 2, 1, 1)
    bias = torch.tensor([1.0, 2.0, 3.0])
    result = probe.output_channel_norms(weight, bias)
    assert set(result) == {'Pedestrian', 'Cyclist', 'Car'}
    assert result['Car']['bias_norm'] == pytest.approx(3.0)


@requires_torch
def test_summarize_gt_counts_uses_kitti_class_order():
    instances = [
        SimpleNamespace(labels_3d=torch.tensor([0, 2, 2])),
        SimpleNamespace(labels_3d=torch.tensor([1, 2])),
    ]
    assert probe.summarize_gt_counts(instances) == {
        'Pedestrian': 1, 'Cyclist': 1, 'Car': 3
    }
