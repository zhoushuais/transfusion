"""Run focused build and one-sample TransFusion KITTI smoke checks."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument(
        '--mode',
        choices=('build', 'loss', 'predict', 'backward'),
        default='backward',
    )
    return parser


def _sum_losses(losses):
    import torch

    values = []
    for name, value in losses.items():
        if not name.startswith('loss') and '_loss_' not in name:
            continue
        if isinstance(value, torch.Tensor):
            values.append(value.mean())
        elif isinstance(value, (list, tuple)):
            values.extend(item.mean() for item in value)
    if not values:
        raise RuntimeError('model returned no differentiable loss tensors')
    return sum(values)


def _one_batch(cfg):
    from mmengine.runner import Runner

    dataloader_cfg = cfg.train_dataloader.copy()
    dataloader_cfg.batch_size = 1
    dataloader_cfg.num_workers = 0
    dataloader_cfg.persistent_workers = False
    dataloader = Runner.build_dataloader(dataloader_cfg)
    return next(iter(dataloader))


def _check_gradients(model, fusion: bool) -> None:
    frozen_with_grad = [
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_with_grad:
        raise RuntimeError(
            f'frozen parameters received gradients: {frozen_with_grad[:5]}')
    trainable_with_grad = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not trainable_with_grad:
        raise RuntimeError('no trainable parameter received a gradient')
    if fusion:
        fusion_prefixes = (
            'bbox_head.shared_conv_img',
            'bbox_head.image_',
            'bbox_head.fusion_prediction_head',
        )
        if not any(
                name.startswith(fusion_prefixes)
                for name in trainable_with_grad):
            raise RuntimeError('no image-fusion parameter received a gradient')


def load_model_checkpoint(model, checkpoint: Path, loader) -> None:
    """Load a checkpoint through MMEngine's string-based path API."""
    loader(model, str(checkpoint), map_location='cpu', strict=False)


def run(args) -> None:
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint
    from mmengine.utils import import_modules_from_strings

    from mmdet3d.registry import MODELS

    cfg = Config.fromfile(str(args.config))
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = MODELS.build(cfg.model)
    model.init_weights()
    checkpoint = args.checkpoint or cfg.get('load_from')
    if checkpoint:
        load_model_checkpoint(model, checkpoint, load_checkpoint)
    model.to(args.device)
    if args.mode == 'build':
        print('MODEL_BUILD_OK')
        return

    raw_batch = _one_batch(cfg)
    training = args.mode in ('loss', 'backward')
    model.train(training)
    batch = model.data_preprocessor(raw_batch, training=training)
    inputs = batch['inputs']
    data_samples = batch['data_samples']

    if args.mode == 'predict':
        with torch.no_grad():
            predictions = model(inputs, data_samples, mode='predict')
        if len(predictions) != 1:
            raise RuntimeError(
                f'expected one prediction, got {len(predictions)}')
        instances = predictions[0].pred_instances_3d
        if instances.bboxes_3d.tensor.shape[-1] != 7:
            raise RuntimeError('KITTI predictions must use seven-value boxes')
        print(f'PREDICT_OK boxes={len(instances.bboxes_3d)}')
        return

    losses = model(inputs, data_samples, mode='loss')
    total_loss = _sum_losses(losses)
    if not torch.isfinite(total_loss):
        raise RuntimeError(f'non-finite total loss: {total_loss.item()}')
    print(f'LOSS_OK total={total_loss.item():.6f}')
    if args.mode == 'backward':
        total_loss.backward()
        _check_gradients(model, fusion=bool(cfg.model.get('fuse_img', False)))
        trainable = sum(parameter.numel() for parameter in model.parameters()
                        if parameter.requires_grad)
        frozen = sum(parameter.numel() for parameter in model.parameters()
                     if not parameter.requires_grad)
        print(f'BACKWARD_OK trainable={trainable} frozen={frozen}')


def main() -> None:
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
