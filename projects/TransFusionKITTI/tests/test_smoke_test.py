from pathlib import Path

from projects.TransFusionKITTI.tools.smoke_test import (build_parser,
                                                        load_model_checkpoint)


def test_smoke_cli_accepts_lidar_config():
    args = build_parser().parse_args([
        'projects/TransFusionKITTI/configs/transfusion_l_kitti.py',
        '--device',
        'cpu',
        '--mode',
        'build',
    ])
    assert args.mode == 'build'


def test_checkpoint_path_is_converted_to_string(tmp_path: Path):
    checkpoint = tmp_path / 'model.pth'
    call = {}

    def loader(model, filename, **kwargs):
        call.update(model=model, filename=filename, kwargs=kwargs)

    model = object()
    load_model_checkpoint(model, checkpoint, loader)

    assert call['model'] is model
    assert call['filename'] == str(checkpoint)
    assert call['kwargs'] == dict(map_location='cpu', strict=False)
