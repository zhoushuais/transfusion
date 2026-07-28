"""Probe Stage-1 TransFusion KITTI heatmap gradients without saving weights."""

import argparse
import copy
import math
from pathlib import Path
import statistics
from typing import Mapping

from projects.TransFusionKITTI.tools.diagnose_stage1 import (
    validate_stage1_config,
)


CLASS_NAMES = ('Pedestrian', 'Cyclist', 'Car')


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


def _summary(values) -> dict:
    values = [float(value) for value in values]
    if not values:
        return dict(count=0, mean=None, min=None, max=None)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError('summary contains NaN or Inf')
    return dict(
        count=len(values),
        mean=float(statistics.fmean(values)),
        min=float(min(values)),
        max=float(max(values)),
    )


def _grouped(values, masks) -> dict:
    flat = values.detach().float().cpu().flatten()
    return dict(
        overall=_summary(flat.tolist()),
        by_class={
            name: _summary(
                values[:, index][masks[:, index]].detach().float().cpu()
                .flatten().tolist())
            for index, name in enumerate(CLASS_NAMES)
        },
    )


def summarize_heatmap(logits, target) -> dict:
    import torch

    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError(
            'heatmap logits and target must share [B,C,H,W] shape')
    if logits.shape[1] != len(CLASS_NAMES):
        raise ValueError('heatmap must contain exactly three KITTI classes')
    probabilities = logits.detach().float().sigmoid()
    positive = target.eq(1)
    background = target.eq(0)
    return dict(
        positive_centers=dict(
            overall=int(positive.sum()),
            by_class={
                name: int(positive[:, index].sum())
                for index, name in enumerate(CLASS_NAMES)
            },
        ),
        positive_probability=_grouped(probabilities, positive),
        background_probability=_grouped(probabilities, background),
        nonfinite_target_count=int((~torch.isfinite(target)).sum()),
        nonfinite_logit_count=int((~torch.isfinite(logits)).sum()),
    )


def parameter_membership(named_parameters: Mapping, optimizer) -> dict:
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    }
    return {
        name: id(parameter) in optimizer_ids
        for name, parameter in named_parameters.items()
    }


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
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_parameters.items()
    }


def summarize_parameter_delta(before: Mapping,
                              named_parameters: Mapping) -> dict:
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
            norm=delta,
            relative=delta / max(base, 1e-12),
        )
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
    if (weight.shape[0] != len(CLASS_NAMES)
            or bias.shape[0] != len(CLASS_NAMES)):
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
