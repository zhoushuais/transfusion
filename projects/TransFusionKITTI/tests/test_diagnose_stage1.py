from pathlib import Path

import pytest

pytest.importorskip('mmengine')

from mmengine.config import ConfigDict

from projects.TransFusionKITTI.tools import diagnose_stage1


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
