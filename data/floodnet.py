# _*_ coding : utf-8 _*_
# @Time: 2024/7/22 9:05
# @Author : Jianchong Guo
# @File : floodnet
# @Project : Knowledge_Distillation_UAV

import warnings

warnings.filterwarnings('ignore')

from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from skimage.io import imread
from collections import OrderedDict
from data.loader import SegmentationLoader
import numpy as np
import logging

logger = logging.getLogger(__name__)

COLOR_MAP = OrderedDict(
    background=(0, 0, 0),
    building_flooded=(255, 0, 0),
    building_no_flooded=(180, 120, 120),
    road_flooded=(160, 150, 20),
    road_non_flooded=(140, 140, 140),
    water=(61, 230, 250),
    tree=(0, 82, 255),
    vehicle=(255, 0, 245),
    pool=(255, 235, 0),
    grass=(4, 250, 7),
)


class FloodNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list = []
        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)
        else:
            self.batch_generate(image_dir, mask_dir)
        self.transforms = transforms

    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

        logger.info('Dataset images: %d' % len(rgb_filepath_list))

        rgb_filename_list = [os.path.split(fp)[-1].split('.')[0] for fp in rgb_filepath_list]

        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                if '_' in fname:
                    cls_filepath_list.append(os.path.join(mask_dir, f'{fname}.png'))
                else:
                    cls_filepath_list.append(os.path.join(mask_dir, f'{fname}_lab.png'))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, item):
        image = imread(self.rgb_filepath_list[item])
        with Image.open(self.cls_filepath_list[item]) as label:
            mask = np.array(label, dtype=np.int64)

        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask)
            image = blob['image']
            mask = blob['mask']

        return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[item]))

    def __len__(self):
        return len(self.rgb_filepath_list)


class FloodNetLoader(SegmentationLoader):
    def __init__(self, config):
        super().__init__(config, FloodNetDataset)
