# _*_ coding : utf-8 _*_
# @Time: 2024/7/23 8:48
# @Author : Jianchong Guo
# @File : function
# @Project : Knowledge_Distillation_UAV

import math

import torch
import torch.nn.functional as F

dim_error = ValueError('image dim should be 3 or 4.')


def th_divisible_pad(tensor, size_divisor: int, mode='constant', value=0):
    """

    Args:
        tensor: 3-D or 4-D tensor of shape [batch (optional), channel, height, width]
        size_divisor: int
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

    Returns:

    """
    if tensor.dim() == 4:
        height, width = tensor.size(2), tensor.size(3)
        tail_pad = [0, 0, 0, 0]
    elif tensor.dim() == 3:
        height, width = tensor.size(1), tensor.size(2)
        tail_pad = [0, 0]
    elif tensor.dim() == 2:
        height, width = tensor.size(0), tensor.size(1)
        tail_pad = []
    else:
        raise dim_error
    nheight = math.ceil(height / size_divisor) * size_divisor
    nwidth = math.ceil(width / size_divisor) * size_divisor
    pad = [0, nwidth - width, 0, nheight - height] + tail_pad

    pad_tensor = F.pad(tensor, pad=pad, mode=mode, value=value)
    return pad_tensor
