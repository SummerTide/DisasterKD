# _*_ coding : utf-8 _*_
# @Time: 2024/8/4 9:58
# @Author : Jianchong Guo
# @File : eval_rescuenet


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
# from data.floodnet import FloodNetLoader, COLOR_MAP
from data.rescuenet import RescueNetLoader, COLOR_MAP
from utils.function import th_divisible_pad
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None):
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis_{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir, palette)

    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)

    model.eval()
    # print(cfg.TEST_DATA_CONFIG)

    test_dataloader = RescueNetLoader(cfg.TEST_DATA_CONFIG)
    metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)

    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            h, w = img.shape[2:4]
            img = F.interpolate(img, size=(672, 896)).cuda()
            pred = model(img)
            if isinstance(pred, list):
                pred = pred[-1].softmax(dim=1)
            pred = F.interpolate(pred, size=(h, w))
            y_true = gt['cls']

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

    cfg = import_config('baseline_rescuenet.deeplabv3_r18')

    model = DeepLabV3(cfg.MODEL)

    model.cuda()

    # ckpt_path = "/data1/gjc23/Code1/Knowledge_Distillation_UAV/logs/rescuenet/pspnet_r18_bs16_60k/RescueNet_PSPNet_r18_60000.pth"
    ckpt_path = "/data1/gjc23/Code1/Knowledge_Distillation_UAV/logs/rescuenet/testkd_pspnet_r101_deeplabv3_r18_bs16_60k_para1/RescueNet_TestKD_PSPNet_r101_DeepLabV3_r18_para1_60000.pth"
    #ckpt_path = "/data1/gjc23/Code1/Knowledge_Distillation_UAV/logs/rescuenet/rescuenet_pspnet_r101_bs16_60k.pth"
    evaluate(model, cfg, False, ckpt_path, logger)

