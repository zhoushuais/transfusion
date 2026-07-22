"""Merge stage-0 image and stage-1 LiDAR checkpoints for TransFusion-LC."""

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        (key[7:] if key.startswith('module.') else key): value
        for key, value in state_dict.items()
    }


def _checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise TypeError(f'{path} does not contain a checkpoint dictionary')
    if 'state_dict' not in checkpoint:
        checkpoint = dict(state_dict=checkpoint)
    if not isinstance(checkpoint['state_dict'], dict):
        raise TypeError(f'{path} state_dict is not a dictionary')
    return checkpoint


def merge(lidar_path: Path, image_path: Path,
          output_path: Path) -> Tuple[dict, Dict[str, object]]:
    """Merge LiDAR state with image backbone/FPN state and save it."""
    lidar_path = Path(lidar_path)
    image_path = Path(image_path)
    output_path = Path(output_path)
    lidar_checkpoint = _checkpoint(lidar_path)
    image_checkpoint = _checkpoint(image_path)
    merged_state = _strip_module_prefix(lidar_checkpoint['state_dict'])
    image_state = _strip_module_prefix(image_checkpoint['state_dict'])

    mapped = []
    skipped = []
    overwritten = []
    for source_key in sorted(image_state):
        if source_key.startswith('backbone.'):
            target_key = f'img_backbone.{source_key[len("backbone."):]}'
        elif source_key.startswith('neck.'):
            target_key = f'img_neck.{source_key[len("neck."):]}'
        else:
            skipped.append(source_key)
            continue
        source_value = image_state[source_key]
        if target_key in merged_state:
            if merged_state[target_key].shape != source_value.shape:
                raise RuntimeError(
                    f'shape collision for {target_key}: '
                    f'{tuple(merged_state[target_key].shape)} vs '
                    f'{tuple(source_value.shape)}')
            overwritten.append(target_key)
        merged_state[target_key] = source_value
        mapped.append((source_key, target_key))

    if not mapped:
        raise RuntimeError('no image backbone or neck keys were mapped')

    merged = dict(lidar_checkpoint)
    merged['state_dict'] = merged_state
    meta = dict(merged.get('meta', {}))
    meta['transfusion_kitti_sources'] = dict(
        lidar=dict(path=str(lidar_path), sha256=_sha256(lidar_path)),
        image=dict(path=str(image_path), sha256=_sha256(image_path)),
    )
    merged['meta'] = meta
    report = dict(
        mapped_image_keys=len(mapped),
        skipped_image_keys=skipped,
        overwritten_keys=overwritten,
        mapped_keys=mapped,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    print(f'mapped image keys: {len(mapped)}')
    print(f'skipped image keys: {len(skipped)}')
    print(f'overwritten keys: {len(overwritten)}')
    print(f'saved: {output_path}')
    return merged, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lidar', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merge(args.lidar, args.image, args.output)


if __name__ == '__main__':
    main()
