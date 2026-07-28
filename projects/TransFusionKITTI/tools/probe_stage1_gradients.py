"""Probe Stage-1 TransFusion KITTI heatmap gradients without saving weights."""

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
from typing import Mapping


def ensure_repository_root_on_path(search_path) -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    repository_root_string = str(repository_root)
    if repository_root_string not in search_path:
        search_path.insert(0, repository_root_string)
    return repository_root


if __package__:
    from .diagnose_stage1 import validate_stage1_config
else:
    import sys

    ensure_repository_root_on_path(sys.path)
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
    parser.add_argument(
        '--override-heatmap-bias',
        type=float,
        default=None,
        help=(
            'diagnostic-only bias for the fresh model heatmap output layer; '
            'the checkpoint model is left unchanged'),
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser


def validate_paths(config: Path, checkpoint: Path, output: Path) -> None:
    if not config.is_file():
        raise FileNotFoundError(f'config does not exist: {config}')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint does not exist: {checkpoint}')
    resolved_output = output.resolve(strict=False)
    protected_paths = {
        config.resolve(strict=False),
        checkpoint.resolve(strict=False),
    }
    if resolved_output in protected_paths:
        raise ValueError('output must not overwrite config or checkpoint')
    if output.suffix.lower() != '.json':
        raise ValueError('output must be a JSON file with a .json suffix')


def validate_probe_config(cfg) -> None:
    validate_stage1_config(cfg)
    if cfg.optim_wrapper.get('type', 'OptimWrapper') != 'OptimWrapper':
        raise ValueError('Stage 1 gradient probe requires FP32 OptimWrapper')
    optimizer_cfg = cfg.optim_wrapper.get('optimizer', {})
    if optimizer_cfg.get('type') != 'AdamW':
        raise ValueError('Stage 1 gradient probe requires AdamW optimizer')
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
    flat = values[masks].detach().float().cpu().flatten()
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


def override_heatmap_bias(model, value: float) -> None:
    """Override only the fresh model dense heatmap output bias for a probe."""
    import torch

    if not math.isfinite(float(value)):
        raise ValueError('--override-heatmap-bias must be finite')
    final_layer = model.bbox_head.heatmap_head[-1]
    if not hasattr(final_layer, 'bias') or final_layer.bias is None:
        raise ValueError('final heatmap layer must have a bias parameter')
    if final_layer.bias.numel() != len(CLASS_NAMES):
        raise ValueError('final heatmap layer must have three output biases')
    with torch.no_grad():
        final_layer.bias.fill_(float(value))


def summarize_gt_counts(gt_instances) -> dict:
    counts = {name: 0 for name in CLASS_NAMES}
    for instances in gt_instances:
        for label in instances.labels_3d.detach().cpu().tolist():
            if label < 0 or label >= len(CLASS_NAMES):
                raise ValueError(f'invalid KITTI class label: {label}')
            counts[CLASS_NAMES[label]] += 1
    return counts


def require_optimizer_state(checkpoint: dict) -> dict:
    optimizer_state = checkpoint.get('optimizer')
    if not isinstance(optimizer_state, dict):
        raise RuntimeError('checkpoint does not contain optimizer state')
    return optimizer_state


def validate_state_result(result: dict) -> None:
    if not all(math.isfinite(value) for value in result['losses'].values()):
        raise RuntimeError('loss contains NaN or Inf')
    if not all(result['requires_grad'].values()):
        raise RuntimeError(
            'heatmap head parameter does not require gradients')
    if not all(result['optimizer_membership'].values()):
        raise RuntimeError('heatmap head parameter is missing from optimizer')
    if result['heatmap']['positive_centers']['overall'] <= 0:
        raise RuntimeError(
            'training batch contains no heatmap positive center')
    if result['heatmap']['nonfinite_target_count']:
        raise RuntimeError('heatmap target contains NaN or Inf')
    if result['heatmap']['nonfinite_logit_count']:
        raise RuntimeError('heatmap logits contain NaN or Inf')
    for key in (
            'heatmap_only_gradients',
            'total_gradients',
            'optimizer_preclip_gradients'):
        if result[key]['nonfinite_count']:
            raise RuntimeError(f'{key} contains NaN or Inf')
    if result['parameter_delta']['nonfinite_count']:
        raise RuntimeError('parameter delta contains NaN or Inf')
    if result['parameter_delta']['all_zero']:
        raise RuntimeError('optimizer step did not update heatmap head')


def _forward_losses(model, raw_batch):
    batch = model.data_preprocessor(copy.deepcopy(raw_batch), training=True)
    inputs = batch['inputs']
    data_samples = batch['data_samples']
    features = model.extract_feat(inputs)
    metas = [sample.metainfo for sample in data_samples]
    predictions = model.bbox_head(features, metas)
    gt_instances = [sample.gt_instances_3d for sample in data_samples]
    targets = model.bbox_head.get_targets(gt_instances, predictions[0])
    dense_logits = predictions[0][0]['dense_heatmap'].clone()
    losses = model.bbox_head.loss_by_feat(predictions, gt_instances)
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
    return {
        name: parameter.grad
        for name, parameter in named_parameters.items()
    }


def _loss_values(losses: dict) -> dict:
    values = {}
    for name, value in losses.items():
        tensors = value if isinstance(value, (list, tuple)) else [value]
        values[name] = float(sum(
            item.detach().float().mean() for item in tensors))
    return values


def _named_final_layer_gradients(named_parameters, gradients, final_layer):
    by_parameter_id = {
        id(parameter): gradient
        for parameter, gradient in zip(named_parameters.values(), gradients)
    }
    weight_gradient = by_parameter_id.get(id(final_layer.weight))
    bias_gradient = by_parameter_id.get(id(final_layer.bias))
    if weight_gradient is None or bias_gradient is None:
        raise RuntimeError('final heatmap layer has no gradient')
    return weight_gradient, bias_gradient


def probe_model_state(model, optim_wrapper, raw_batch) -> dict:
    import torch

    model.train()
    optim_wrapper.zero_grad()
    losses, dense_logits, target, gt_counts = _forward_losses(
        model, raw_batch)
    total_loss, _ = model.parse_losses(losses)
    loss_values = dict(total=float(total_loss.detach()),
                       **_loss_values(losses))
    heatmap = summarize_heatmap(dense_logits, target)
    named_parameters = _heatmap_parameters(model)
    requires_grad = {
        name: parameter.requires_grad
        for name, parameter in named_parameters.items()
    }
    optimizer_membership = parameter_membership(
        named_parameters, optim_wrapper.optimizer)
    preflight = dict(
        losses=loss_values,
        requires_grad=requires_grad,
        optimizer_membership=optimizer_membership,
        heatmap=heatmap,
        heatmap_only_gradients=dict(nonfinite_count=0),
        total_gradients=dict(nonfinite_count=0),
        optimizer_preclip_gradients=dict(nonfinite_count=0),
        parameter_delta=dict(nonfinite_count=0, all_zero=False),
    )
    validate_state_result(preflight)

    parameters = tuple(named_parameters.values())
    isolated = torch.autograd.grad(
        losses['loss_heatmap'],
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    isolated_by_name = dict(zip(named_parameters, isolated))
    heatmap_only = summarize_gradients(isolated_by_name)

    scaled_total_loss = optim_wrapper.scale_loss(total_loss)
    optim_wrapper.backward(scaled_total_loss)
    total_gradients = summarize_gradients(
        _current_gradients(named_parameters))
    all_optimizer_gradients = {
        f'group_{group_index}.{parameter_index}': parameter.grad
        for group_index, group in enumerate(
            optim_wrapper.optimizer.param_groups)
        for parameter_index, parameter in enumerate(group['params'])
        if parameter.requires_grad
    }
    preclip = summarize_gradients(all_optimizer_gradients)

    final_layer = model.bbox_head.heatmap_head[-1]
    isolated_weight, isolated_bias = _named_final_layer_gradients(
        named_parameters, isolated, final_layer)
    total_weight, total_bias = _named_final_layer_gradients(
        named_parameters,
        tuple(parameter.grad for parameter in named_parameters.values()),
        final_layer,
    )
    final_bias = [
        float(value) for value in final_layer.bias.detach().float().cpu()
    ]
    output_channel_heatmap_only = output_channel_norms(
        isolated_weight, isolated_bias)
    output_channel_total = output_channel_norms(total_weight, total_bias)

    before = snapshot_parameters(named_parameters)
    learning_rates = [
        float(group['lr'])
        for group in optim_wrapper.optimizer.param_groups
    ]
    optim_wrapper.step()
    delta = summarize_parameter_delta(before, named_parameters)
    optim_wrapper.zero_grad()

    result = dict(
        gt_counts=gt_counts,
        losses=loss_values,
        heatmap=heatmap,
        final_bias=final_bias,
        requires_grad=requires_grad,
        optimizer_membership=optimizer_membership,
        heatmap_only_gradients=heatmap_only,
        total_gradients=total_gradients,
        output_channel_heatmap_only_gradients=(
            output_channel_heatmap_only),
        output_channel_total_gradients=output_channel_total,
        optimizer_preclip_gradients=preclip,
        learning_rates=learning_rates,
        parameter_delta=delta,
    )
    validate_state_result(result)
    return result


def _safe_ratio(numerator: float, denominator: float):
    if denominator == 0:
        return None
    return float(numerator / denominator)


def build_comparison(fresh: dict, checkpoint: dict) -> dict:
    fresh_probability = fresh[
        'heatmap']['positive_probability']['overall']['mean']
    checkpoint_probability = checkpoint[
        'heatmap']['positive_probability']['overall']['mean']
    fresh_gradient = fresh['heatmap_only_gradients']['overall_norm']
    checkpoint_gradient = checkpoint[
        'heatmap_only_gradients']['overall_norm']
    return dict(
        positive_probability_ratio_checkpoint_to_fresh=_safe_ratio(
            checkpoint_probability, fresh_probability),
        positive_probability_change_checkpoint_minus_fresh=float(
            checkpoint_probability - fresh_probability),
        heatmap_gradient_ratio_checkpoint_to_fresh=_safe_ratio(
            checkpoint_gradient, fresh_gradient),
        heatmap_gradient_change_checkpoint_minus_fresh=float(
            checkpoint_gradient - fresh_gradient),
        both_have_gradients_and_update=(
            fresh_gradient > 0
            and checkpoint_gradient > 0
            and not fresh['parameter_delta']['all_zero']
            and not checkpoint['parameter_delta']['all_zero']
        ),
        checkpoint_low_positive_response=(
            checkpoint_probability is not None
            and checkpoint_probability < 0.05
        ),
    )


def build_batch_summary(fresh: dict, checkpoint: dict) -> dict:
    fresh_centers = fresh['heatmap']['positive_centers']
    checkpoint_centers = checkpoint['heatmap']['positive_centers']
    if (fresh['gt_counts'] != checkpoint['gt_counts']
            or fresh_centers != checkpoint_centers):
        raise RuntimeError('fresh and checkpoint batch targets differ')
    return dict(
        gt_counts=fresh['gt_counts'],
        positive_centers=fresh_centers,
    )


def write_report(result: dict, output: Path) -> None:
    serialized = json.dumps(result, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding='utf-8')


def print_summary(result: dict, output: Path) -> None:
    for name in ('fresh', 'checkpoint'):
        state = result[name]
        positive = state[
            'heatmap']['positive_probability']['overall']['mean']
        print(
            f'{name}: loss_heatmap={state["losses"]["loss_heatmap"]:.6f} '
            f'positive_probability={positive:.6f} '
            f'heatmap_grad='
            f'{state["heatmap_only_gradients"]["overall_norm"]:.6f} '
            f'parameter_delta='
            f'{state["parameter_delta"]["overall_norm"]:.6e}')
    print('comparison:', result['comparison'])
    print(f'STAGE1_GRADIENT_PROBE_OK output={output}')


def _build_model_and_optimizer(cfg,
                               device,
                               checkpoint_path=None,
                               heatmap_bias=None):
    from mmengine.optim import build_optim_wrapper
    from mmengine.runner.checkpoint import load_checkpoint

    from mmdet3d.registry import MODELS

    model = MODELS.build(cfg.model)
    model.init_weights()
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(
            model,
            str(checkpoint_path),
            map_location='cpu',
            strict=True,
        )
    if heatmap_bias is not None:
        override_heatmap_bias(model, heatmap_bias)
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

    validate_paths(args.config, args.checkpoint, args.output)
    cfg = Config.fromfile(str(args.config))
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    validate_probe_config(cfg)

    set_random_seed(args.seed, deterministic=False)
    dataloader = Runner.build_dataloader(
        build_probe_dataloader_cfg(cfg),
        seed=args.seed,
        diff_rank_seed=False,
    )
    raw_batch = next(iter(dataloader))
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    fresh_model, fresh_optimizer = _build_model_and_optimizer(
        cfg,
        device,
        heatmap_bias=args.override_heatmap_bias,
    )
    fresh = probe_model_state(fresh_model, fresh_optimizer, raw_batch)
    del fresh_model, fresh_optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    checkpoint_model, checkpoint_optimizer = _build_model_and_optimizer(
        cfg, device, args.checkpoint)
    checkpoint = probe_model_state(
        checkpoint_model, checkpoint_optimizer, raw_batch)

    result = dict(
        metadata=dict(
            config=str(args.config.resolve()),
            checkpoint=str(args.checkpoint.resolve()),
            device=str(device),
            seed=args.seed,
            batch_size=int(cfg.train_dataloader.batch_size),
            accumulative_counts=int(
                cfg.optim_wrapper.get('accumulative_counts', 1)),
            override_heatmap_bias=args.override_heatmap_bias,
            class_names=list(CLASS_NAMES),
        ),
        batch=build_batch_summary(fresh, checkpoint),
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
