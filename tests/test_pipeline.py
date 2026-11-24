import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from disasterkd.cli import main
from disasterkd.prepare import prepare, tile_offsets


def write_pair(raw, split, number, size, label):
    image_dir = raw / split / f'{split}-org-img'
    mask_dir = raw / split / f'{split}-label-img'
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, (37, 79, 161)).save(image_dir / f'{number}.jpg')
    Image.new('L', size, label).save(mask_dir / f'{number}_lab.png')


class PipelineTest(unittest.TestCase):
    def test_end_to_end_without_downloads(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, data, runs = root / 'raw', root / 'datasets', root / 'runs'
            write_pair(raw, 'train', 1, (1100, 1024), 1)
            write_pair(raw, 'test', 2, (67, 61), 2)
            main(['prepare', '--dataset', 'floodnet', '--raw-root', str(raw), '--data-root', str(data)])
            self.assertEqual(tile_offsets(3000), [0, 1024, 1976])
            self.assertEqual(len(list((data / 'floodnet/train/images').glob('*.jpg'))), 2)
            shared = ['--dataset', 'floodnet', '--data-root', str(data), '--run-root', str(runs), '--device', 'cpu']
            short = ['--workers', '0', '--steps', '1', '--no-pretrained', '--crop-size', '64', '--batch-size', '2']
            main(['teacher', *shared, *short])
            main(['distill', *shared, *short])
            main(['evaluate', *shared])
            result = runs / 'floodnet/disasterkd/pspnet18'
            self.assertTrue((result / 'student.pth').is_file())
            self.assertEqual(json.loads((result / 'metrics.json').read_text())['evaluated_pixels'], 67 * 61)

    def test_rescuenet_v1_labels_survive_resizing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / 'raw'
            write_pair(raw, 'train', 1, (130, 99), 11)
            write_pair(raw, 'test', 2, (80, 45), 0)
            prepare('rescuenet', raw, root / 'data')
            with Image.open(root / 'data/rescuenet/train/masks/1_lab.png') as mask:
                self.assertEqual(mask.size, (896, 672))
                self.assertEqual(np.asarray(mask).min(), 11)
            write_pair(raw, 'train', 1, (130, 99), 10)
            with self.assertRaisesRegex(ValueError, 'V1 masks'):
                prepare('rescuenet', raw, root / 'other')
            self.assertFalse((root / 'other/rescuenet').exists())
