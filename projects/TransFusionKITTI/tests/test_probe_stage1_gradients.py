# flake8: noqa: E402
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
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
    assert args.override_heatmap_bias is None


def test_script_help_runs_from_repository_root():
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / (
        'projects/TransFusionKITTI/tools/probe_stage1_gradients.py')
    result = subprocess.run(
        [sys.executable, str(script), '--help'],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '--checkpoint' in result.stdout


def test_readme_gradient_probe_uses_resumable_epoch_checkpoint():
    repository_root = Path(__file__).resolve().parents[3]
    readme = (repository_root / 'projects/TransFusionKITTI/README.md').read_text(
        encoding='utf-8')
    gradient_probe_section = readme.split(
        '### Stage 1 gradient probe', maxsplit=1)[1]
    assert ('--checkpoint '
            'work_dirs/transfusion_l_kitti_formal_run1_xyfix/epoch_5.pth'
            in gradient_probe_section)
    assert 'best_Kitti metric_pred_instances_3d' not in gradient_probe_section


def test_ensure_repository_root_on_path_prepends_root():
    search_path = ['existing']
    root = probe.ensure_repository_root_on_path(search_path)
    assert root == Path(__file__).resolve().parents[3]
    assert search_path == [str(root), 'existing']
    probe.ensure_repository_root_on_path(search_path)
    assert search_path.count(str(root)) == 1


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
    cfg.optim_wrapper.type = 'OptimWrapper'
    cfg.optim_wrapper.optimizer.type = 'SGD'
    with pytest.raises(ValueError, match='AdamW'):
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
        probe.validate_paths(
            config, tmp_path / 'missing.pth', tmp_path / 'probe.json')


def test_validate_paths_rejects_unsafe_output(tmp_path: Path):
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_5.pth'
    config.write_text('model = dict()', encoding='utf-8')
    checkpoint.write_text('checkpoint', encoding='utf-8')
    with pytest.raises(ValueError, match='must not overwrite'):
        probe.validate_paths(config, checkpoint, config)
    with pytest.raises(ValueError, match='must not overwrite'):
        probe.validate_paths(config, checkpoint, checkpoint)
    with pytest.raises(ValueError, match='JSON'):
        probe.validate_paths(config, checkpoint, tmp_path / 'probe.txt')
    probe.validate_paths(config, checkpoint, tmp_path / 'probe.json')


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


def test_grouped_overall_uses_only_masked_values():
    class ArrayTensor:

        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, index):
            if isinstance(index, ArrayTensor):
                index = index.values
            return ArrayTensor(self.values[index])

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def flatten(self):
            return ArrayTensor(self.values.flatten())

        def tolist(self):
            return self.values.tolist()

    values = ArrayTensor([[
        [[0.8, 0.2]],
        [[0.3, 0.4]],
        [[0.5, 0.6]],
    ]])
    masks = ArrayTensor([[
        [[True, False]],
        [[False, False]],
        [[False, True]],
    ]])
    result = probe._grouped(values, masks)
    assert result['overall']['count'] == 2
    assert result['overall']['mean'] == pytest.approx(0.7)
    assert result['by_class']['Pedestrian']['mean'] == pytest.approx(0.8)
    assert result['by_class']['Cyclist']['count'] == 0
    assert result['by_class']['Car']['mean'] == pytest.approx(0.6)


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
def test_override_heatmap_bias_changes_only_three_output_biases():
    final_layer = torch.nn.Conv2d(4, 3, kernel_size=1)
    model = SimpleNamespace(
        bbox_head=SimpleNamespace(
            heatmap_head=torch.nn.Sequential(final_layer)))
    probe.override_heatmap_bias(model, -2.19)
    torch.testing.assert_close(
        final_layer.bias.detach(), torch.full((3,), -2.19))


@requires_torch
def test_override_heatmap_bias_rejects_nonfinite_value():
    final_layer = torch.nn.Conv2d(4, 3, kernel_size=1)
    model = SimpleNamespace(
        bbox_head=SimpleNamespace(
            heatmap_head=torch.nn.Sequential(final_layer)))
    with pytest.raises(ValueError, match='finite'):
        probe.override_heatmap_bias(model, float('nan'))


@requires_torch
def test_summarize_gt_counts_uses_kitti_class_order():
    instances = [
        SimpleNamespace(labels_3d=torch.tensor([0, 2, 2])),
        SimpleNamespace(labels_3d=torch.tensor([1, 2])),
    ]
    assert probe.summarize_gt_counts(instances) == {
        'Pedestrian': 1, 'Cyclist': 1, 'Car': 3
    }


def test_require_optimizer_state_rejects_missing_state():
    with pytest.raises(RuntimeError, match='optimizer state'):
        probe.require_optimizer_state({'state_dict': {}})
    state = {'state': {}, 'param_groups': []}
    assert probe.require_optimizer_state({'optimizer': state}) is state


def _valid_state_result():
    return dict(
        losses=dict(total=1.0, loss_heatmap=0.5),
        requires_grad={'weight': True},
        optimizer_membership={'weight': True},
        heatmap_only_gradients=dict(nonfinite_count=0, overall_norm=1.0),
        total_gradients=dict(nonfinite_count=0, overall_norm=1.0),
        optimizer_preclip_gradients=dict(
            nonfinite_count=0, overall_norm=1.0),
        parameter_delta=dict(
            all_zero=False, overall_norm=0.1, nonfinite_count=0),
        heatmap=dict(
            nonfinite_target_count=0,
            nonfinite_logit_count=0,
            positive_centers=dict(overall=1),
        ),
    )


def test_validate_state_result_rejects_zero_delta():
    valid = _valid_state_result()
    probe.validate_state_result(valid)
    invalid = copy.deepcopy(valid)
    invalid['parameter_delta']['all_zero'] = True
    with pytest.raises(RuntimeError, match='did not update'):
        probe.validate_state_result(invalid)


@pytest.mark.parametrize(
    ('path', 'value', 'message'),
    [
        (('requires_grad', 'weight'), False, 'does not require gradients'),
        (('losses', 'total'), float('nan'), 'loss contains'),
        (('parameter_delta', 'nonfinite_count'), 1,
         'parameter delta contains'),
        (('optimizer_preclip_gradients', 'nonfinite_count'), 1,
         'optimizer_preclip_gradients contains'),
    ],
)
def test_validate_state_result_rejects_invalid_numeric_or_grad_state(
        path, value, message):
    invalid = _valid_state_result()
    invalid[path[0]][path[1]] = value
    with pytest.raises(RuntimeError, match=message):
        probe.validate_state_result(invalid)


def _state(positive, gradient, delta):
    return dict(
        heatmap=dict(positive_probability=dict(overall=dict(mean=positive))),
        heatmap_only_gradients=dict(overall_norm=gradient),
        parameter_delta=dict(all_zero=not delta),
    )


def test_build_comparison_reports_both_update_paths():
    result = probe.build_comparison(
        _state(0.5, 4.0, True),
        _state(0.03, 1.0, True),
    )
    assert result['both_have_gradients_and_update'] is True
    assert result[
        'positive_probability_ratio_checkpoint_to_fresh'] == pytest.approx(
            0.06)
    assert result[
        'heatmap_gradient_ratio_checkpoint_to_fresh'] == pytest.approx(0.25)


def test_write_report_rejects_nan(tmp_path: Path):
    output = tmp_path / 'probe.json'
    with pytest.raises(ValueError):
        probe.write_report({'value': float('nan')}, output)
    assert not output.exists()


def test_build_batch_summary_requires_matching_targets():
    fresh = dict(
        gt_counts={'Pedestrian': 1, 'Cyclist': 2, 'Car': 3},
        heatmap=dict(positive_centers=dict(
            overall=6,
            by_class={'Pedestrian': 1, 'Cyclist': 2, 'Car': 3},
        )),
    )
    checkpoint = copy.deepcopy(fresh)
    assert probe.build_batch_summary(fresh, checkpoint) == {
        'gt_counts': fresh['gt_counts'],
        'positive_centers': fresh['heatmap']['positive_centers'],
    }
    checkpoint['heatmap']['positive_centers']['overall'] = 5
    with pytest.raises(RuntimeError, match='batch targets differ'):
        probe.build_batch_summary(fresh, checkpoint)


def test_forward_losses_preserves_logits_before_inplace_loss_mutation():
    class EmptyLabels:

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return []

    class MutableHeatmap:

        def __init__(self, value):
            self.value = value

        def clone(self):
            return MutableHeatmap(self.value)

    class FakeHead:

        def __init__(self):
            self.heatmap = MutableHeatmap(2.0)

        def __call__(self, features, metas):
            return [[{'dense_heatmap': self.heatmap}]]

        def get_targets(self, gt_instances, predictions):
            return ('target',)

        def loss_by_feat(self, predictions, gt_instances):
            predictions[0][0]['dense_heatmap'].value = 0.75
            return {'loss_heatmap': 1.0}

    class FakeModel:

        def __init__(self):
            self.bbox_head = FakeHead()

        def data_preprocessor(self, raw_batch, training):
            sample = SimpleNamespace(
                metainfo={},
                gt_instances_3d=SimpleNamespace(labels_3d=EmptyLabels()),
            )
            return {'inputs': 'inputs', 'data_samples': [sample]}

        def extract_feat(self, inputs):
            return 'features'

    model = FakeModel()
    _, dense_logits, _, _ = probe._forward_losses(model, {})
    assert model.bbox_head.heatmap.value == 0.75
    assert dense_logits.value == 2.0
