# _*_ coding : utf-8 _*_
# @Time: 2024/8/4 9:58
# @Author : Jianchong Guo
# @File : eval_rescuenet


import logging
import os

import numpy as np
import torch
from utils.metrics import PixelMetric
from tqdm import tqdm

from utils.tools import count_model_parameters
from utils.visualize import VisualizeSegmm
from data.rescuenet import RescueNetLoader, COLOR_MAP
import torch.nn.functional as F

def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None):
    logger = logger or logging.getLogger(__name__)
    device = next(model.parameters()).device
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis_{}'.format(os.path.basename(ckpt_path) if ckpt_path else 'current'))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir, palette)

    if not is_training:
        model_state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(model_state_dict)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)

    model.eval()

    test_dataloader = RescueNetLoader(cfg.TEST_DATA_CONFIG)
    metric_op = PixelMetric(len(COLOR_MAP), logdir=cfg.SNAPSHOT_DIR, logger=logger)

    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            h, w = img.shape[2:4]
            img = F.interpolate(img, size=(672, 896)).to(device)
            pred = model(img)
            if isinstance(pred, list):
                pred = pred[-1].softmax(dim=1)
            pred = F.interpolate(pred, size=(h, w))
            y_true = gt['cls']

            pred = pred.argmax(dim=1).cpu()

            valid_inds = y_true != -1
            metric_op.update(y_true[valid_inds], pred[valid_inds])

            if cfg.SNAPSHOT_DIR is not None:
                for clsmap, imname in zip(pred, gt['fname']):
                    viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('jpg', 'png'))

    torch.cuda.empty_cache()
    return metric_op.summary_all()

if __name__ == '__main__':
    from utils.evaluation import run_evaluation

    run_evaluation(evaluate, 'baseline_rescuenet.deeplabv3_r18',
                   './logs/rescuenet/testkd_pspnet_r101_deeplabv3_r18_bs16_60k_para1/RescueNet_TestKD_PSPNet_r101_DeepLabV3_r18_para1_60000.pth')
