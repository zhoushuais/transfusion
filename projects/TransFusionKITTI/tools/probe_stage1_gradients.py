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
