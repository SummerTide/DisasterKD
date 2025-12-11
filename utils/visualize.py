# _*_ coding : utf-8 _*_
# @Time: 2024/7/22 17:27
# @Author : Jianchong Guo
# @File : visualize
# @Project : Knowledge_Distillation_UAV

import os
import numpy as np
from PIL import Image


def put_palette(mask, palette):
    mask = Image.fromarray(mask)
    mask.putpalette(palette)
    mask = mask.convert('RGB')
    mask = np.asarray(mask)
    return mask


class VisualizeSegmm(object):
    def __init__(self, out_dir, palette):
        self.out_dir = out_dir
        self.palette = palette
        os.makedirs(self.out_dir, exist_ok=True)

    def __call__(self, y_pred, filename):
        y_pred = y_pred.astype(np.uint8)
        y_pred = y_pred.squeeze()
        color_y = Image.fromarray(y_pred)
        color_y.putpalette(self.palette)
        color_y.save(os.path.join(self.out_dir, filename))