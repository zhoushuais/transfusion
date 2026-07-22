from importlib import import_module


def test_project_package_imports():
    module = import_module('projects.TransFusionKITTI.transfusion_kitti')
    assert module.__all__ == []
