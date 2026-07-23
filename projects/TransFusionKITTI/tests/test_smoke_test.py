from pathlib import Path
from types import SimpleNamespace

from projects.TransFusionKITTI.tools import smoke_test
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


def test_select_device_sets_current_cuda_device(monkeypatch):
    selected = []
    device = SimpleNamespace(type='cuda', index=2)
    fake_torch = SimpleNamespace(
        device=lambda value: device,
        cuda=SimpleNamespace(set_device=selected.append),
    )
    monkeypatch.setitem(__import__('sys').modules, 'torch', fake_torch)

    result = smoke_test.select_device('cuda:2')

    assert result is device
    assert selected == [device]
