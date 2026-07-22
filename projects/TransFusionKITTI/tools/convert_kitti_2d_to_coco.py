"""Convert KITTI ``label_2`` annotations to COCO detection JSON."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image

CLASS_NAMES = ('Pedestrian', 'Cyclist', 'Car')
CATEGORY_IDS = {name: index + 1 for index, name in enumerate(CLASS_NAMES)}


def _sample_ids(split_file: Path) -> List[str]:
    sample_ids = [
        line.strip()
        for line in split_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f'duplicate sample id in {split_file}')
    return sample_ids


def _parse_annotation(line: str, width: int, height: int,
                      sample_id: str) -> Dict:
    fields = line.split()
    if len(fields) < 15:
        raise ValueError(
            f'sample {sample_id}: expected at least 15 KITTI label fields')
    class_name = fields[0]
    if class_name not in CATEGORY_IDS:
        return {}
    try:
        x1, y1, x2, y2 = (float(value) for value in fields[4:8])
    except ValueError as error:
        raise ValueError(f'sample {sample_id}: invalid 2D bbox') from error
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return {}
    return dict(
        category_id=CATEGORY_IDS[class_name],
        bbox=[x1, y1, box_width, box_height],
        area=box_width * box_height,
        iscrowd=0,
    )


def convert(root: Path, split_file: Path, output: Path) -> None:
    """Convert one KITTI split to a deterministic COCO annotation file."""
    root = Path(root)
    split_file = Path(split_file)
    output = Path(output)
    images = []
    annotations = []
    annotation_id = 1

    for image_id, sample_id in enumerate(_sample_ids(split_file), start=1):
        image_path = root / 'training' / 'image_2' / f'{sample_id}.png'
        label_path = root / 'training' / 'label_2' / f'{sample_id}.txt'
        if not image_path.is_file():
            raise FileNotFoundError(
                f'sample {sample_id}: missing {image_path}')
        if not label_path.is_file():
            raise FileNotFoundError(
                f'sample {sample_id}: missing {label_path}')
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            dict(
                id=image_id,
                file_name=f'training/image_2/{sample_id}.png',
                width=width,
                height=height,
            ))
        for line in label_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            annotation = _parse_annotation(line, width, height, sample_id)
            if not annotation:
                continue
            annotation.update(id=annotation_id, image_id=image_id)
            annotations.append(annotation)
            annotation_id += 1

    payload = dict(
        images=images,
        annotations=annotations,
        categories=[
            dict(id=CATEGORY_IDS[name], name=name) for name in CLASS_NAMES
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + '\n',
        encoding='utf-8',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert(args.root, args.split, args.output)


if __name__ == '__main__':
    main()
