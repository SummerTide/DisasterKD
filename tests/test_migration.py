import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from module.baseline.resnet import ResNetEncoder
from utils.metrics import PixelMetric


class MigrationTest(unittest.TestCase):
    def test_encoder_checkpoint_keys_and_receptive_field(self):
        # A strict load alone cannot detect incorrect convolution strides/dilations.
        for name, final_channels in [('resnet18', 512), ('resnet101', 2048)]:
            with self.subTest(name=name):
                encoder = ResNetEncoder(resnet_type=name, output_stride=8).eval()
                state = encoder.state_dict()
                self.assertIn('resnet.layer4.0.downsample.0.weight', state)
                self.assertNotIn('resnet.fc.weight', state)
                layer = encoder.resnet.layer4[0]
                transition = layer.conv1 if name == 'resnet18' else layer.conv2
                self.assertEqual(transition.stride, (1, 1))
                self.assertEqual(transition.dilation, (2, 2))
                self.assertEqual(encoder.resnet.layer4[1].conv2.dilation, (4, 4))
                with torch.no_grad():
                    stages = encoder(torch.zeros(1, 3, 32, 64))
                self.assertEqual([x.shape[-1] for x in stages], [16, 8, 8, 8])
                self.assertEqual(stages[-1].shape, (1, final_channels, 4, 8))

    def test_pixel_scores_across_batches_include_absent_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            metric = PixelMetric(3, logdir=directory)
            metric.update([0, 0], [0, 1])
            metric.update([1], [1])
            result = metric.summary_all()
            np.testing.assert_equal(result['iou'], [0.5, 0.5, 0.0])
            self.assertEqual(result['mIoU'], 0.33333)
            self.assertEqual(result['OA'], 0.66667)
            cm = np.load(next((Path(directory) / 'cm').glob('*.npy')))
            np.testing.assert_equal(cm, [[1, 1, 0], [0, 1, 0], [0, 0, 0]])


if __name__ == '__main__':
    unittest.main()
