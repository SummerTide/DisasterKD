"""Prepare the original supervised train/test splits without changing the source archive."""

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def tile_offsets(length, size=1024):
    if length < size:
        raise ValueError(f"FloodNet training sides must be at least {size} pixels, got {length}")
    return [min(start, length - size) for start in range(0, length, size)]


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare(dataset, raw_root, data_root):
    raw_root = Path(raw_root).expanduser()
    data_root = Path(data_root).expanduser()
    destination = data_root / dataset
    if destination.exists():
        raise FileExistsError(f"{destination} already exists. Reuse it for training or choose another --data-root.")
    data_root.mkdir(parents=True, exist_ok=True)
    max_id = 9 if dataset == 'floodnet' else 11
    seen_max = 0
    with TemporaryDirectory(dir=data_root) as tmp:
        staged = Path(tmp) / dataset
        counts = {}
        for split in ('train', 'test'):
            source = raw_root / split
            images = sorted((source / f'{split}-org-img').glob('*.jpg'))
            if not images:
                raise ValueError(f"No JPEGs in {source}/{split}-org-img. Pass the directory containing train/ and test/.")
            out_images, out_masks = staged / split / 'images', staged / split / 'masks'
            out_images.mkdir(parents=True)
            out_masks.mkdir()
            count = 0
            for image_path in tqdm(images, desc=f'{dataset} {split}'):
                mask_path = source / f'{split}-label-img' / f'{image_path.stem}_lab.png'
                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    if mask.mode not in ('L', 'P') or mask.size != image.size:
                        raise ValueError(f"Expected an indexed mask matching image dimensions: {mask_path}")
                    largest = mask.getextrema()[1]
                    if largest > max_id:
                        raise ValueError(f"Unexpected label {largest} in {mask_path}")
                    seen_max = max(seen_max, largest)
                    if split == 'test':
                        link_or_copy(image_path, out_images / image_path.name)
                        link_or_copy(mask_path, out_masks / mask_path.name)
                        count += 1
                    elif dataset == 'floodnet':
                        for y in tile_offsets(image.height):
                            for x in tile_offsets(image.width):
                                box = (x, y, x + 1024, y + 1024)
                                stem = image_path.stem + '_' + '_'.join(map(str, box))
                                image.crop(box).convert('RGB').save(out_images / (stem + '.jpg'), quality=75)
                                mask.crop(box).save(out_masks / (stem + '.png'))
                                count += 1
                    else:
                        rgb = cv2.resize(cv2.imread(str(image_path)), (896, 672), interpolation=cv2.INTER_CUBIC)
                        target = cv2.resize(np.asarray(mask), (896, 672), interpolation=cv2.INTER_NEAREST)
                        if not cv2.imwrite(str(out_images / image_path.name), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise OSError(f'Failed to write resized image {image_path.name}')
                        Image.fromarray(target).save(out_masks / mask_path.name)
                        count += 1
            counts[split] = count
        if dataset == 'rescuenet' and seen_max < 11:
            note = raw_root / 'RescueNet-DATASET-VERSION-NOTE-v1.0.txt'
            if not note.is_file() or 'Sand' not in note.read_text(errors='replace'):
                raise ValueError("RescueNet needs the V1 masks with Debris and Sand (ID 11). "
                                 "For a V1 subset, include the original V1 version note.")
        staged.rename(destination)
    print(f"Prepared {destination}: {counts['train']} training pairs, {counts['test']} test pairs")
