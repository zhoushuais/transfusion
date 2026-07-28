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
