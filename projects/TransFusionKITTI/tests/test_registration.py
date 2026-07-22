from importlib import import_module

import pytest

pytest.importorskip('torch')
pytest.importorskip('mmcv')
pytest.importorskip('mmengine')
pytest.importorskip('mmdet')


def test_project_package_imports():
    module = import_module('projects.TransFusionKITTI.transfusion_kitti')
    assert 'TransFusionKITTIBBoxCoder' in module.__all__
