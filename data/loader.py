"""Shared data loading for step-based segmentation experiments."""

import torch
from albumentations import Compose, HorizontalFlip, Normalize, OneOf, RandomRotate90, VerticalFlip
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset


def default_transforms():
    return Compose([
        OneOf([HorizontalFlip(p=1), VerticalFlip(p=1), RandomRotate90(p=1)], p=0.75),
        Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1),
        ToTensorV2(),
    ])


class SegmentationLoader(DataLoader):
    def __init__(self, config, dataset_type):
        transforms = config['transforms'] if 'transforms' in config else default_transforms()
        dataset = dataset_type(config['image_dir'], config['mask_dir'], transforms)
        training = config.get('training', True)
        cv = config.get('CV', {})
        fold = cv.get('i', -1)
        if fold != -1:
            k = cv['k']
            if not (0 <= fold < k and 2 <= k <= len(dataset)):
                raise ValueError("CV requires 2 <= k <= dataset size and 0 <= i < k")
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(2333)).tolist()
            if training:
                indices = [idx for pos, idx in enumerate(indices) if pos % k != fold]
            else:
                indices = indices[fold::k]
            dataset = Subset(dataset, indices)

        super().__init__(dataset, batch_size=config.get('batch_size', 4), shuffle=training,
                         num_workers=config.get('num_workers', 4), pin_memory=True)


def repeat_batches(loader):
    """Start a fresh sampling pass each epoch without caching batches."""
    if not len(loader):
        raise ValueError("Cannot repeat an empty data loader")
    while True:
        yield from loader
