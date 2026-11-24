import os
from pathlib import Path


def split_dirs(dataset, split, root=None):
    root = Path(root or os.environ.get('DISASTERKD_DATA_ROOT', 'datasets')).expanduser()
    directory = root / dataset / split
    return dict(image_dir=str(directory / 'images'), mask_dir=str(directory / 'masks'))
