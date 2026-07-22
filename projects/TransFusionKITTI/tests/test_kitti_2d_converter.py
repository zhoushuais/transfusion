import json
from pathlib import Path

import numpy as np
from PIL import Image

from projects.TransFusionKITTI.tools.convert_kitti_2d_to_coco import convert


def test_converter_filters_dontcare_and_keeps_three_classes(tmp_path: Path):
    root = tmp_path / 'kitti'
    (root / 'training/image_2').mkdir(parents=True)
    (root / 'training/label_2').mkdir(parents=True)
    (root / 'ImageSets').mkdir()
    Image.fromarray(np.zeros(
        (100, 200, 3),
        dtype=np.uint8)).save(root / 'training/image_2/000001.png')
    (root / 'training/label_2/000001.txt').write_text(
        'Pedestrian 0 0 0 10 20 30 60 1 1 1 1 1 1 0\n'
        'DontCare -1 -1 -10 0 0 40 40 -1 -1 -1 -1000 -1000 -1000 -10\n',
        encoding='utf-8',
    )
    (root / 'ImageSets/train.txt').write_text('000001\n', encoding='utf-8')
    output = root / 'annotations/kitti_2d_train.json'
    convert(root, root / 'ImageSets/train.txt', output)
    data = json.loads(output.read_text(encoding='utf-8'))
    assert [category['name'] for category in data['categories']] == [
        'Pedestrian',
        'Cyclist',
        'Car',
    ]
    assert len(data['annotations']) == 1
    assert data['annotations'][0]['bbox'] == [10.0, 20.0, 20.0, 40.0]


def test_converter_clips_boxes_and_rejects_malformed_labels(tmp_path: Path):
    root = tmp_path / 'kitti'
    (root / 'training/image_2').mkdir(parents=True)
    (root / 'training/label_2').mkdir(parents=True)
    Image.fromarray(np.zeros(
        (20, 30, 3),
        dtype=np.uint8)).save(root / 'training/image_2/000002.png')
    (root / 'training/label_2/000002.txt').write_text(
        'Car 0 0 0 -5 -2 40 22 1 1 1 1 1 1 0\n', encoding='utf-8')
    split = root / 'split.txt'
    split.write_text('000002\n', encoding='utf-8')
    output = root / 'out.json'
    convert(root, split, output)
    annotation = json.loads(
        output.read_text(encoding='utf-8'))['annotations'][0]
    assert annotation['bbox'] == [0.0, 0.0, 30.0, 20.0]
