# _*_ coding : utf-8 _*_
# @Time: 2024/9/5 14:28
# @Author : Jianchong Guo
# @File : eval_floodnet

import logging
import math
import numpy as np
from torch import nn
import ever as er
from tqdm import tqdm

from module.baseline.deeplabv3 import DeepLabV3
from module.baseline.pspnet import PSPNet
from utils.tools import *
from ever.util.param_util import count_model_parameters
from utils.visualize import VisualizeSegmm
from torch.nn.modules.utils import _pair
from data.floodnet import FloodNetLoader, COLOR_MAP
from utils.function import th_divisible_pad

logger = logging.getLogger(__name__)


def sliding_window(input_size, kernel_size, stride):
    ih, iw = input_size
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    assert ih > 0 and iw > 0 and kh > 0 and kw > 0 and sh > 0 and sw > 0

    kh = ih if kh > ih else kh
    kw = iw if kw > iw else kw

    num_rows = math.ceil((ih - kh) / sh) if math.ceil((ih - kh) / sh) * sh + kh >= ih else math.ceil(
        (ih - kh) / sh) + 1
    num_cols = math.ceil((iw - kw) / sw) if math.ceil((iw - kw) / sw) * sw + kw >= iw else math.ceil(
        (iw - kw) / sw) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * sw
    ymin = y * sh

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + kw > iw, iw - xmin - kw, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + kh > ih, ih - ymin - kh, np.zeros_like(ymin))
    boxes = np.stack([xmin + xmin_offset, ymin + ymin_offset,
                      np.minimum(xmin + kw, iw), np.minimum(ymin + kh, ih)], axis=1)

    return boxes


class SegmSlidingWinInference(object):
    def __init__(self):
        super(SegmSlidingWinInference, self).__init__()
        self._h = None
        self._w = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def patch(self, input_size, patch_size, stride, transforms=None):
        """ divide large image into small patches.

        Returns:

        """
        self.wins = sliding_window(input_size, patch_size, stride)
        self.transforms = transforms
        return self

    def merge(self, out_list):
        pred_list, win_list = list(zip(*out_list))
        num_classes = pred_list[0].size(1)
        res_img = torch.zeros(pred_list[0].size(0), num_classes, self._h, self._w, dtype=torch.float32)
        res_count = torch.zeros(self._h, self._w, dtype=torch.float32)

        for pred, win in zip(pred_list, win_list):
            res_count[win[1]:win[3], win[0]: win[2]] += 1
            res_img[:, :, win[1]:win[3], win[0]: win[2]] += pred.cpu()

        avg_res_img = res_img / res_count

        return avg_res_img

    def forward(self, model, image_tensor, **kwargs):
        assert self.wins is not None, 'patch must be performed before forward.'
        # set the image height and width
        self._h, self._w = image_tensor.shape[2:4]
        return self._forward(model, image_tensor, **kwargs)

    def _forward(self, model, image_tensor, **kwargs):
        self.device = kwargs.get('device', self.device)
        size_divisor = kwargs.get('size_divisor', None)
        assert self.wins is not None, 'patch must be performed before forward.'
        out_list = []
        for win in tqdm(self.wins):
            x1, y1, x2, y2 = win
            image = image_tensor[:, :, y1:y2, x1:x2]
            if self.transforms is not None:
                image = self.transforms(image=image)['image']
            h, w = image.shape[2:4]
            if size_divisor is not None:
                image = th_divisible_pad(image, size_divisor)
            image = image.to(self.device)
            with torch.no_grad():
                out = model(image)
            if isinstance(out, list):
                out = out[-1].softmax(dim=1)
            if size_divisor is not None:
                out = out[:, :, :h, :w]
            out_list.append((out.cpu(), win))
            torch.cuda.empty_cache()
        self.wins = None

        return self.merge(out_list)


def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None):
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis_{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir, palette)

    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict, strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)

    model.eval()
    # print(cfg.TEST_DATA_CONFIG)

    test_dataloader = FloodNetLoader(cfg.TEST_DATA_CONFIG)
    metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)
    segm_helper = SegmSlidingWinInference()

    with torch.no_grad():
        for idx, (img, gt) in enumerate(test_dataloader):
            h, w = img.shape[2:4]
            logging.info('Progress - [{} / {}] size = ({}, {})'.format(idx + 1, len(test_dataloader), h, w))
            seg_helper = segm_helper.patch((h, w), patch_size=(1024, 1024), stride=512, transforms=None)

            pred = seg_helper.forward(model, img, size_divisor=32)

            y_true = gt['cls']
            y_true = y_true.cpu()

            pred = pred.argmax(dim=1).cpu()

            valid_inds = y_true != -1
            metric_op.forward(y_true[valid_inds], pred[valid_inds])

            if cfg.SNAPSHOT_DIR is not None:
                for clsmap, imname in zip(pred, gt['fname']):
                    viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('jpg', 'png'))

    metric_op.summary_all()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    seed_torch(2333)

    cfg = import_config('baseline_floodnet.deeplabv3_r101')

    model = PSPNet(cfg.MODEL)

    model.cuda()

    ckpt_path = "/data1/gjc23/Code/Knowledge_Distillation_UAV/logs/floodnet/floodnet_deeplabv3_r101_bs16_60k.pth"
    evaluate(model, cfg, False, ckpt_path, logger)
