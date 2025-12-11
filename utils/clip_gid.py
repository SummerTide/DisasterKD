# _*_ coding : utf-8 _*_
# @Time: 2024/7/22 10:02
# @Author : Jianchong Guo
# @File : clip_gid
# @Project : Knowledge_Distillation_UAV

import glob
import os
from skimage.io import imsave, imread
import numpy as np
from tqdm import tqdm
import math
from torch.nn.modules.utils import _pair


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


# imgp_list = glob.glob(os.path.join('/data1/gjc23/Dataset/RescueNet/train-org-img', '*.jpg'))  # image
imgp_list = glob.glob(os.path.join('/data1/gjc23/Dataset/RescueNet/train-label-img', '*.png'))  # label
patch_size = 1024
stride = 1024
save_dir = r'/data1/gjc23/Dataset/RescueNet_p1024/train/label_p1024s1024'
os.makedirs(save_dir, exist_ok=True)

for imgp in tqdm(imgp_list):
    raw_image = imread(imgp)
    fname = os.path.basename(imgp)
    # h, w, _ = raw_image.shape
    h, w = raw_image.shape  # label
    boxes = sliding_window((h, w), patch_size, stride)
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        clipped_image = raw_image[y_min:y_max, x_min:x_max].astype(np.uint8)
        # save_path = os.path.join(save_dir, '%s_%d_%d_%d_%d.jpg' % (fname.replace('.jpg', ''), x_min, y_min, x_max, y_max))  # image
        save_path = os.path.join(save_dir, '%s_%d_%d_%d_%d.png' % (fname.replace('_lab.png', ''), x_min, y_min, x_max, y_max))  # label
        imsave(save_path, clipped_image, check_contrast=False)
        
        
        
