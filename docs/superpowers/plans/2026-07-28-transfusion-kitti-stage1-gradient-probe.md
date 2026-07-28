# TransFusion KITTI Stage 1 Gradient Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个不会保存 checkpoint 的 Stage 1 梯度探针，用同一训练 batch 比较随机初始化模型与 epoch 5 checkpoint 的 heatmap target、梯度、optimizer 归属和参数更新。

**Architecture:** 新工具沿用 `diagnose_stage1.py` 的 Stage 1 配置约束，但将所有可单测的统计、参数快照和结果比较函数保留在独立脚本中。生产路径按配置构建 train dataloader、模型和 `OptimWrapper`，复用一个 CPU raw batch，分别探测 fresh/checkpoint 状态，并将事实型结果写入 JSON。

**Tech Stack:** Python 3.8、PyTorch 2.1.2、MMEngine 0.10.7、MMCV 2.1.0、MMDetection 3.2.0、MMDetection3D 1.4.0、pytest。

---

## File Structure

- Create: `projects/TransFusionKITTI/tools/probe_stage1_gradients.py`
  - CLI、train dataloader 收敛、heatmap/梯度/参数统计、两个模型状态的生产探针和 JSON 报告。
- Create: `projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py`
  - 不依赖 KITTI 的纯函数与轻量 PyTorch 单测；真实 CUDA 链路留给 H800。
- Modify: `projects/TransFusionKITTI/README.md`
  - 在 Stage 1 故障诊断段落补充单次探针命令、输出和判读边界。

现有模型、配置、训练入口和 `diagnose_stage1.py` 不修改。

### Task 1: Define CLI, Config, And Data Contracts

**Files:**
- Create: `projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py`
- Create: `projects/TransFusionKITTI/tools/probe_stage1_gradients.py`

- [ ] **Step 1: Write the failing contract tests**

Create the test file with:

```python
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
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
```

Expected: collection fails because `probe_stage1_gradients.py` does not exist.

- [ ] **Step 3: Implement the CLI and configuration boundary**

Create `probe_stage1_gradients.py` with this initial content:

```python
"""Probe Stage-1 TransFusion KITTI heatmap gradients without saving weights."""

import argparse
import copy
from pathlib import Path

from projects.TransFusionKITTI.tools.diagnose_stage1 import (
    validate_stage1_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def validate_paths(config: Path, checkpoint: Path) -> None:
    if not config.is_file():
        raise FileNotFoundError(f'config does not exist: {config}')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint does not exist: {checkpoint}')


def validate_probe_config(cfg) -> None:
    validate_stage1_config(cfg)
    if cfg.optim_wrapper.get('type', 'OptimWrapper') != 'OptimWrapper':
        raise ValueError('Stage 1 gradient probe requires FP32 OptimWrapper')
    if int(cfg.train_dataloader.get('batch_size', -1)) <= 0:
        raise ValueError('train_dataloader.batch_size must be positive')
    if int(cfg.optim_wrapper.get('accumulative_counts', 1)) <= 0:
        raise ValueError('accumulative_counts must be positive')


def build_probe_dataloader_cfg(cfg):
    dataloader_cfg = copy.deepcopy(cfg.train_dataloader)
    dataloader_cfg.num_workers = 0
    dataloader_cfg.persistent_workers = False
    dataloader_cfg.drop_last = False
    dataloader_cfg.sampler.shuffle = False
    return dataloader_cfg
```

- [ ] **Step 4: Run contract tests**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
```

Expected: `4 passed` when PyTorch is available, or config-only tests pass with explicitly marked PyTorch skips later.

- [ ] **Step 5: Commit the contract**

```bash
git add projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py \
  projects/TransFusionKITTI/tools/probe_stage1_gradients.py
git commit -m "test: define Stage 1 gradient probe contracts"
```

### Task 2: Add Heatmap, Gradient, And Parameter Metrics

**Files:**
- Modify: `projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py`
- Modify: `projects/TransFusionKITTI/tools/probe_stage1_gradients.py`

- [ ] **Step 1: Write failing metric tests**

Append:

```python
try:
    import torch
except ImportError:
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason='requires PyTorch')


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
    assert result['positive_probability']['overall']['mean'] == pytest.approx(0.5)
    assert result['nonfinite_target_count'] == 0


@requires_torch
def test_parameter_membership_reports_missing_parameter():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.SGD(model[0].parameters(), lr=0.1)
    result = probe.parameter_membership(dict(model.named_parameters()), optimizer)
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
```

- [ ] **Step 2: Run metric tests and verify missing helpers**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
```

Expected: failures name `summarize_heatmap`, `parameter_membership`, `summarize_gradients`, `snapshot_parameters`, `summarize_parameter_delta`, `output_channel_norms`, and `summarize_gt_counts`.

- [ ] **Step 3: Implement JSON-safe metric helpers**

Add imports and helpers:

```python
import math
import statistics
from typing import Mapping

CLASS_NAMES = ('Pedestrian', 'Cyclist', 'Car')


def _summary(values) -> dict:
    values = [float(value) for value in values]
    if not values:
        return dict(count=0, mean=None, min=None, max=None)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError('summary contains NaN or Inf')
    return dict(
        count=len(values), mean=float(statistics.fmean(values)),
        min=float(min(values)), max=float(max(values)))


def _grouped(values, masks) -> dict:
    flat = values.detach().float().cpu().flatten()
    return dict(
        overall=_summary(flat.tolist()),
        by_class={
            name: _summary(values[:, index][masks[:, index]].detach().float()
                           .cpu().flatten().tolist())
            for index, name in enumerate(CLASS_NAMES)
        },
    )


def summarize_heatmap(logits, target) -> dict:
    import torch
    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError('heatmap logits and target must share [B,C,H,W] shape')
    if logits.shape[1] != len(CLASS_NAMES):
        raise ValueError('heatmap must contain exactly three KITTI classes')
    probabilities = logits.detach().float().sigmoid()
    positive = target.eq(1)
    background = target.eq(0)
    return dict(
        positive_centers=dict(
            overall=int(positive.sum()),
            by_class={name: int(positive[:, index].sum())
                      for index, name in enumerate(CLASS_NAMES)}),
        positive_probability=_grouped(probabilities, positive),
        background_probability=_grouped(probabilities, background),
        nonfinite_target_count=int((~torch.isfinite(target)).sum()),
        nonfinite_logit_count=int((~torch.isfinite(logits)).sum()),
    )


def parameter_membership(named_parameters: Mapping, optimizer) -> dict:
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups
        for parameter in group['params']
    }
    return {name: id(parameter) in optimizer_ids
            for name, parameter in named_parameters.items()}


def summarize_gradients(named_gradients: Mapping) -> dict:
    import torch
    squared = 0.0
    nonfinite = 0
    by_parameter = {}
    for name, gradient in named_gradients.items():
        if gradient is None:
            by_parameter[name] = None
            continue
        detached = gradient.detach().float()
        nonfinite += int((~torch.isfinite(detached)).sum())
        norm = float(torch.linalg.vector_norm(detached))
        by_parameter[name] = norm
        squared += norm * norm
    return dict(
        overall_norm=math.sqrt(squared),
        nonfinite_count=nonfinite,
        by_parameter=by_parameter,
    )


def snapshot_parameters(named_parameters: Mapping) -> dict:
    return {name: parameter.detach().cpu().clone()
            for name, parameter in named_parameters.items()}


def summarize_parameter_delta(before: Mapping, named_parameters: Mapping) -> dict:
    import torch

    squared = 0.0
    base_squared = 0.0
    nonfinite = 0
    by_parameter = {}
    for name, parameter in named_parameters.items():
        current = parameter.detach().float().cpu()
        previous = before[name].float()
        nonfinite += int((~torch.isfinite(current)).sum())
        nonfinite += int((~torch.isfinite(previous)).sum())
        delta = float((current - previous).norm())
        base = float(previous.norm())
        by_parameter[name] = dict(
            norm=delta, relative=delta / max(base, 1e-12))
        squared += delta * delta
        base_squared += base * base
    overall = math.sqrt(squared)
    return dict(
        overall_norm=overall,
        relative_norm=overall / max(math.sqrt(base_squared), 1e-12),
        nonfinite_count=nonfinite,
        all_zero=overall == 0.0,
        by_parameter=by_parameter,
    )


def output_channel_norms(weight, bias) -> dict:
    if weight.shape[0] != len(CLASS_NAMES) or bias.shape[0] != len(CLASS_NAMES):
        raise ValueError('final heatmap layer must have three output channels')
    return {
        name: dict(
            weight_norm=float(weight[index].detach().float().norm()),
            bias_norm=float(bias[index].detach().float().abs()),
        )
        for index, name in enumerate(CLASS_NAMES)
    }


def summarize_gt_counts(gt_instances) -> dict:
    counts = {name: 0 for name in CLASS_NAMES}
    for instances in gt_instances:
        for label in instances.labels_3d.detach().cpu().tolist():
            if label < 0 or label >= len(CLASS_NAMES):
                raise ValueError(f'invalid KITTI class label: {label}')
            counts[CLASS_NAMES[label]] += 1
    return counts
```

- [ ] **Step 4: Run metric tests**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
```

Expected: all available tests pass; PyTorch-only tests may skip on the local Windows runtime.

- [ ] **Step 5: Commit metric primitives**

```bash
git add projects/TransFusionKITTI/tools/probe_stage1_gradients.py \
  projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
git commit -m "feat: add Stage 1 gradient probe metrics"
```

### Task 3: Implement One-State Production Gradient Probe

**Files:**
- Modify: `projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py`
- Modify: `projects/TransFusionKITTI/tools/probe_stage1_gradients.py`

- [ ] **Step 1: Write failing state-validation tests**

Append:

```python
def test_require_optimizer_state_rejects_missing_state():
    with pytest.raises(RuntimeError, match='optimizer state'):
        probe.require_optimizer_state({'state_dict': {}})
    state = {'state': {}, 'param_groups': []}
    assert probe.require_optimizer_state({'optimizer': state}) is state


@requires_torch
def test_validate_probe_result_rejects_zero_delta_and_nonfinite_gradient():
    valid = dict(
        losses=dict(total=1.0, loss_heatmap=0.5),
        requires_grad={'weight': True},
        optimizer_membership={'weight': True},
        heatmap_only_gradients=dict(nonfinite_count=0, overall_norm=1.0),
        total_gradients=dict(nonfinite_count=0, overall_norm=1.0),
        parameter_delta=dict(
            all_zero=False, overall_norm=0.1, nonfinite_count=0),
        heatmap=dict(nonfinite_target_count=0, nonfinite_logit_count=0,
                     positive_centers=dict(overall=1)),
    )
    probe.validate_state_result(valid)
    invalid = copy.deepcopy(valid)
    invalid['parameter_delta']['all_zero'] = True
    with pytest.raises(RuntimeError, match='did not update'):
        probe.validate_state_result(invalid)
```

Add `import copy` to the test file.

- [ ] **Step 2: Run the tests and verify validation helpers are missing**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
```

Expected: failures name `require_optimizer_state` and `validate_state_result`.

- [ ] **Step 3: Implement checkpoint and result validation**

Add:

```python
def require_optimizer_state(checkpoint: dict) -> dict:
    optimizer_state = checkpoint.get('optimizer')
    if not isinstance(optimizer_state, dict):
        raise RuntimeError('checkpoint does not contain optimizer state')
    return optimizer_state


def validate_state_result(result: dict) -> None:
    if not all(math.isfinite(value) for value in result['losses'].values()):
        raise RuntimeError('loss contains NaN or Inf')
    if not all(result['requires_grad'].values()):
        raise RuntimeError('heatmap head parameter does not require gradients')
    if not all(result['optimizer_membership'].values()):
        raise RuntimeError('heatmap head parameter is missing from optimizer')
    if result['heatmap']['positive_centers']['overall'] <= 0:
        raise RuntimeError('training batch contains no heatmap positive center')
    if result['heatmap']['nonfinite_target_count']:
        raise RuntimeError('heatmap target contains NaN or Inf')
    if result['heatmap']['nonfinite_logit_count']:
        raise RuntimeError('heatmap logits contain NaN or Inf')
    for key in ('heatmap_only_gradients', 'total_gradients'):
        if result[key]['nonfinite_count']:
            raise RuntimeError(f'{key} contains NaN or Inf')
    if result['parameter_delta']['nonfinite_count']:
        raise RuntimeError('parameter delta contains NaN or Inf')
    if result['parameter_delta']['all_zero']:
        raise RuntimeError('optimizer step did not update heatmap head')
```

- [ ] **Step 4: Implement production forward and one-state probe**

Add these functions, importing PyTorch/MMEngine inside production functions so CLI/config tests remain importable without CUDA:

```python
def _forward_losses(model, raw_batch):
    batch = model.data_preprocessor(copy.deepcopy(raw_batch), training=True)
    inputs = batch['inputs']
    data_samples = batch['data_samples']
    features = model.extract_feat(inputs)
    metas = [sample.metainfo for sample in data_samples]
    predictions = model.bbox_head(features, metas)
    gt_instances = [sample.gt_instances_3d for sample in data_samples]
    targets = model.bbox_head.get_targets(gt_instances, predictions[0])
    losses = model.bbox_head.loss_by_feat(predictions, gt_instances)
    dense_logits = predictions[0][0]['dense_heatmap']
    return (
        losses,
        dense_logits,
        targets[-1],
        summarize_gt_counts(gt_instances),
    )


def _heatmap_parameters(model) -> dict:
    parameters = dict(model.bbox_head.heatmap_head.named_parameters())
    if not parameters:
        raise RuntimeError('heatmap head has no parameters')
    return parameters


def _current_gradients(named_parameters: Mapping) -> dict:
    return {name: parameter.grad for name, parameter in named_parameters.items()}


def _loss_values(losses: dict) -> dict:
    values = {}
    for name, value in losses.items():
        tensors = value if isinstance(value, (list, tuple)) else [value]
        values[name] = float(sum(item.detach().float().mean()
                                 for item in tensors))
    return values


def probe_model_state(model, optim_wrapper, raw_batch) -> dict:
    import torch

    model.train()
    optim_wrapper.zero_grad()
    losses, dense_logits, target, gt_counts = _forward_losses(
        model, raw_batch)
    total_loss, _ = model.parse_losses(losses)
    loss_heatmap = losses['loss_heatmap']
    named_parameters = _heatmap_parameters(model)
    parameters = tuple(named_parameters.values())
    isolated = torch.autograd.grad(
        loss_heatmap, parameters, retain_graph=True, allow_unused=True)
    heatmap_only = summarize_gradients(dict(zip(named_parameters, isolated)))

    scaled_total_loss = optim_wrapper.scale_loss(total_loss)
    optim_wrapper.backward(scaled_total_loss)
    total_gradients = summarize_gradients(_current_gradients(named_parameters))
    all_optimizer_gradients = {
        f'group_{group_index}.{parameter_index}': parameter.grad
        for group_index, group in enumerate(optim_wrapper.optimizer.param_groups)
        for parameter_index, parameter in enumerate(group['params'])
        if parameter.requires_grad
    }
    preclip = summarize_gradients(all_optimizer_gradients)
    before = snapshot_parameters(named_parameters)
    learning_rates = [float(group['lr'])
                      for group in optim_wrapper.optimizer.param_groups]
    optim_wrapper.step()
    delta = summarize_parameter_delta(before, named_parameters)
    optim_wrapper.zero_grad()

    final_layer = model.bbox_head.heatmap_head[-1]
    result = dict(
        gt_counts=gt_counts,
        losses=dict(total=float(total_loss.detach()), **_loss_values(losses)),
        heatmap=summarize_heatmap(dense_logits, target),
        final_bias=[float(value) for value in final_layer.bias.detach().cpu()],
        requires_grad={name: parameter.requires_grad
                       for name, parameter in named_parameters.items()},
        optimizer_membership=parameter_membership(
            named_parameters, optim_wrapper.optimizer),
        heatmap_only_gradients=heatmap_only,
        total_gradients=total_gradients,
        output_channel_heatmap_only_gradients=output_channel_norms(
            isolated[-2], isolated[-1]),
        output_channel_total_gradients=output_channel_norms(
            final_layer.weight.grad, final_layer.bias.grad),
        optimizer_preclip_gradients=preclip,
        learning_rates=learning_rates,
        parameter_delta=delta,
    )
    validate_state_result(result)
    return result
```

Before zeroing gradients, store final-layer total gradient tensors locally; otherwise the output-channel summary must be computed before `optim_wrapper.zero_grad()`. Keep that ordering explicit in the implementation.

- [ ] **Step 5: Run focused tests and syntax compilation**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
python -m py_compile projects/TransFusionKITTI/tools/probe_stage1_gradients.py
```

Expected: available tests pass and compilation exits 0.

- [ ] **Step 6: Commit the production state probe**

```bash
git add projects/TransFusionKITTI/tools/probe_stage1_gradients.py \
  projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
git commit -m "feat: probe Stage 1 heatmap gradients"
```

### Task 4: Build Fresh/Checkpoint Comparison And CLI

**Files:**
- Modify: `projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py`
- Modify: `projects/TransFusionKITTI/tools/probe_stage1_gradients.py`

- [ ] **Step 1: Write failing comparison and JSON tests**

Append:

```python
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
    assert result['positive_probability_ratio_checkpoint_to_fresh'] == pytest.approx(0.06)
    assert result['heatmap_gradient_ratio_checkpoint_to_fresh'] == pytest.approx(0.25)


def test_write_report_rejects_nan(tmp_path: Path):
    output = tmp_path / 'probe.json'
    with pytest.raises(ValueError):
        probe.write_report({'value': float('nan')}, output)
    assert not output.exists()
```

- [ ] **Step 2: Run tests and verify comparison/report helpers are missing**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
```

Expected: failures name `build_comparison` and `write_report`.

- [ ] **Step 3: Implement comparison, report, and terminal summary**

Add:

```python
import json


def _safe_ratio(numerator: float, denominator: float):
    if denominator == 0:
        return None
    return float(numerator / denominator)


def build_comparison(fresh: dict, checkpoint: dict) -> dict:
    fresh_probability = fresh['heatmap']['positive_probability']['overall']['mean']
    checkpoint_probability = checkpoint['heatmap']['positive_probability']['overall']['mean']
    fresh_gradient = fresh['heatmap_only_gradients']['overall_norm']
    checkpoint_gradient = checkpoint['heatmap_only_gradients']['overall_norm']
    return dict(
        positive_probability_ratio_checkpoint_to_fresh=_safe_ratio(
            checkpoint_probability, fresh_probability),
        heatmap_gradient_ratio_checkpoint_to_fresh=_safe_ratio(
            checkpoint_gradient, fresh_gradient),
        both_have_gradients_and_update=(
            fresh_gradient > 0 and checkpoint_gradient > 0 and
            not fresh['parameter_delta']['all_zero'] and
            not checkpoint['parameter_delta']['all_zero']),
        checkpoint_low_positive_response=(
            checkpoint_probability is not None and
            checkpoint_probability < 0.05),
    )


def write_report(result: dict, output: Path) -> None:
    serialized = json.dumps(result, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding='utf-8')


def print_summary(result: dict, output: Path) -> None:
    for name in ('fresh', 'checkpoint'):
        state = result[name]
        print(
            f'{name}: loss_heatmap={state["losses"]["loss_heatmap"]:.6f} '
            f'positive_probability='
            f'{state["heatmap"]["positive_probability"]["overall"]["mean"]:.6f} '
            f'heatmap_grad={state["heatmap_only_gradients"]["overall_norm"]:.6f} '
            f'parameter_delta={state["parameter_delta"]["overall_norm"]:.6e}')
    print('comparison:', result['comparison'])
    print(f'STAGE1_GRADIENT_PROBE_OK output={output}')
```

- [ ] **Step 4: Implement model construction and `run`**

Add:

```python
def _build_model_and_optimizer(cfg, device, checkpoint_path=None):
    from mmengine.optim import build_optim_wrapper
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet3d.registry import MODELS

    model = MODELS.build(cfg.model)
    model.init_weights()
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(
            model, str(checkpoint_path), map_location='cpu', strict=True)
    model.to(device)
    optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
    if checkpoint is not None:
        optimizer_state = copy.deepcopy(require_optimizer_state(checkpoint))
        optim_wrapper.load_state_dict(optimizer_state)
    return model, optim_wrapper


def run(args) -> dict:
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner, set_random_seed
    from mmengine.utils import import_modules_from_strings

    validate_paths(args.config, args.checkpoint)
    cfg = Config.fromfile(str(args.config))
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    validate_probe_config(cfg)

    set_random_seed(args.seed, deterministic=False)
    dataloader = Runner.build_dataloader(
        build_probe_dataloader_cfg(cfg), seed=args.seed,
        diff_rank_seed=False)
    raw_batch = next(iter(dataloader))
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    fresh_model, fresh_optimizer = _build_model_and_optimizer(cfg, device)
    fresh = probe_model_state(fresh_model, fresh_optimizer, raw_batch)
    del fresh_model, fresh_optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    checkpoint_model, checkpoint_optimizer = _build_model_and_optimizer(
        cfg, device, args.checkpoint)
    checkpoint = probe_model_state(
        checkpoint_model, checkpoint_optimizer, raw_batch)

    if (fresh['gt_counts'] != checkpoint['gt_counts'] or
            fresh['heatmap']['positive_centers'] !=
            checkpoint['heatmap']['positive_centers']):
        raise RuntimeError('fresh and checkpoint batch targets differ')

    result = dict(
        metadata=dict(
            config=str(args.config.resolve()),
            checkpoint=str(args.checkpoint.resolve()),
            device=str(device), seed=args.seed,
            batch_size=int(cfg.train_dataloader.batch_size),
            accumulative_counts=int(
                cfg.optim_wrapper.get('accumulative_counts', 1)),
            class_names=list(CLASS_NAMES),
        ),
        batch=dict(
            gt_counts=fresh['gt_counts'],
            positive_centers=fresh['heatmap']['positive_centers'],
        ),
        fresh=fresh,
        checkpoint=checkpoint,
        comparison=build_comparison(fresh, checkpoint),
    )
    write_report(result, args.output)
    print_summary(result, args.output)
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
```

The final implementation must set `fresh_model.train()` and checkpoint model training mode only after loading parameters, and it must not call any runner/checkpoint save API.

- [ ] **Step 5: Run focused tests and compilation**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
python -m py_compile projects/TransFusionKITTI/tools/probe_stage1_gradients.py
```

Expected: all available tests pass and compilation exits 0.

- [ ] **Step 6: Commit the CLI**

```bash
git add projects/TransFusionKITTI/tools/probe_stage1_gradients.py \
  projects/TransFusionKITTI/tests/test_probe_stage1_gradients.py
git commit -m "feat: add Stage 1 gradient probe CLI"
```

### Task 5: Document, Verify, And Push

**Files:**
- Modify: `projects/TransFusionKITTI/README.md`

- [ ] **Step 1: Add the H800 command and safety boundary**

Add a section after the existing read-only Stage 1 diagnostics:

```markdown
### Stage 1 gradient probe

Only run this after `stage1_diagnostics.json` shows low dense GT-center scores.
It compares a fresh model and the selected Stage 1 checkpoint on the same
training batch. It performs one diagnostic optimizer step in memory but never
saves a model checkpoint.

```bash
CUDA_VISIBLE_DEVICES=2 python \
  projects/TransFusionKITTI/tools/probe_stage1_gradients.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --checkpoint "work_dirs/transfusion_l_kitti_formal_run1_xyfix/best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_epoch_5.pth" \
  --output work_dirs/transfusion_l_kitti_formal_run1_xyfix/stage1_gradient_probe.json
```

Success is marked by `STAGE1_GRADIENT_PROBE_OK`. This only proves that the
diagnostic completed; it does not prove Stage 1 is fixed. Do not start Stage 2
or rerun the 40-epoch Stage 1 before reviewing the JSON.
```

- [ ] **Step 2: Run project tests**

Run:

```bash
pytest -q projects/TransFusionKITTI/tests
```

Expected on the current local runtime: all non-PyTorch tests pass and PyTorch/CUDA tests are explicitly skipped. Record the exact counts rather than assuming them.

- [ ] **Step 3: Run static verification**

Run:

```bash
python -m py_compile projects/TransFusionKITTI/tools/probe_stage1_gradients.py
git diff --check
git status --short
```

Expected: compile and diff checks exit 0; status contains only intended README/script/test changes before commit.

- [ ] **Step 4: Commit documentation**

```bash
git add projects/TransFusionKITTI/README.md
git commit -m "docs: document Stage 1 gradient probe"
```

- [ ] **Step 5: Review the complete branch diff**

Run:

```bash
git diff --check origin/codex/transfusion-kitti-port..HEAD
git log --oneline origin/codex/transfusion-kitti-port..HEAD
git status --short --branch
```

Expected: clean worktree; branch contains the design, plan, tested tool, tests, and README changes only.

- [ ] **Step 6: Push the completed branch**

```bash
git push origin codex/transfusion-kitti-port
```

Expected: remote branch advances successfully. H800 numerical validation remains pending until the user manually downloads the updated ZIP and runs the single documented command.
