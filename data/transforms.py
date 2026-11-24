# _*_ coding : utf-8 _*_
# @Time: 2024/7/22 9:33
# @Author : Jianchong Guo
# @File : transforms
# @Project : Knowledge_Distillation_UAV

import random
import numpy as np
from albumentations import DualTransform, RandomScale
import math

__all__ = [
    'DivisiblePad', 'RandomDiscreteScale',
]


class RandomDiscreteScale(RandomScale):
    """Sample one of the experiment's scale factors for both image and mask."""

    def __init__(self, scales, p=0.5):
        super().__init__(scale_limit=0, p=p)
        self.scales = tuple(scales)

    def get_params(self):
        return {"scale": random.choice(self.scales)}

    def get_transform_init_args_names(self):
        return ("scales",)


class DivisiblePad(DualTransform):
    def __init__(self,
                 size_divisor=32,
                 value=0,
                 mask_value=-1,
                 always_apply=False,
                 p=1.0):
        super(DivisiblePad, self).__init__(always_apply, p)
        self.size_divisor = size_divisor
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, **params):
        height, width, _ = img.shape

        nheight = math.ceil(height / self.size_divisor) * self.size_divisor
        nwidth = math.ceil(width / self.size_divisor) * self.size_divisor
        pad_bottom = nwidth - width
        pad_right = nheight - height
        return np.pad(img, ((0, pad_right), (0, pad_bottom), (0, 0)), mode='constant',
                      constant_values=self.value)

    def apply_to_mask(self, img, **params):
        height, width = img.shape

        nheight = math.ceil(height / self.size_divisor) * self.size_divisor
        nwidth = math.ceil(width / self.size_divisor) * self.size_divisor
        pad_bottom = nwidth - width
        pad_right = nheight - height

        return np.pad(img, ((0, pad_right), (0, pad_bottom)), mode='constant',
                      constant_values=self.mask_value)

    def get_transform_init_args_names(self):
        return "size_divisor", "value", "mask_value"


if __name__ == '__main__':
    x = np.ones((1024, 1024, 3))
    mask = np.ones((1024, 1024))
    o = DivisiblePad()(image=x, mask=mask)
    print(o['image'].shape)
    print(o['mask'].shape)
