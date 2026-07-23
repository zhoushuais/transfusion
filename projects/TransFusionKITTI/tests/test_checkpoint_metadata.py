import importlib
import sys
from collections import OrderedDict
from types import ModuleType
from unittest.mock import patch


def _load_merge_module_without_torch():
    module_name = 'projects.TransFusionKITTI.tools.merge_pretrained_weights'
    existing_module = sys.modules.pop(module_name, None)
    try:
        with patch.dict(sys.modules, {'torch': ModuleType('torch')}):
            module = importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if existing_module is not None:
            sys.modules[module_name] = existing_module
    return module


def test_strip_module_prefix_preserves_spconv_metadata():
    merge_module = _load_merge_module_without_torch()
    state = OrderedDict({
        'module.middle_encoder.conv_input.0.weight': object()
    })
    state._metadata = OrderedDict({
        '': {
            'version': 1
        },
        'module': {
            'version': 1
        },
        'module.middle_encoder': {
            'version': 2
        },
        'module.middle_encoder.conv_input': {
            'version': 2
        },
    })

    stripped = merge_module._strip_module_prefix(state)

    assert isinstance(stripped, OrderedDict)
    assert list(stripped) == ['middle_encoder.conv_input.0.weight']
    assert stripped._metadata['middle_encoder']['version'] == 2
    assert stripped._metadata['middle_encoder.conv_input']['version'] == 2


def test_merge_image_metadata_remaps_backbone_and_neck():
    merge_module = _load_merge_module_without_torch()
    merged_state = OrderedDict()
    merged_state._metadata = OrderedDict({
        'middle_encoder': {
            'version': 2
        }
    })
    image_state = OrderedDict()
    image_state._metadata = OrderedDict({
        '': {
            'version': 1
        },
        'backbone': {
            'version': 3
        },
        'backbone.conv': {
            'version': 4
        },
        'neck': {
            'version': 5
        },
        'roi_head': {
            'version': 6
        },
    })

    merge_module._merge_image_metadata(merged_state, image_state)

    assert merged_state._metadata['middle_encoder']['version'] == 2
    assert merged_state._metadata['img_backbone']['version'] == 3
    assert merged_state._metadata['img_backbone.conv']['version'] == 4
    assert merged_state._metadata['img_neck']['version'] == 5
    assert 'img_roi_head' not in merged_state._metadata
