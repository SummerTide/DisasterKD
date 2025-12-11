# _*_ coding : utf-8 _*_
# @Time: 2024/8/2 8:14
# @Author : Jianchong Guo
# @File : tools
# @Project : Knowledge_Distillation_UAV

import importlib
import logging
import time
import os
import torch.nn.functional as F
import torch
import shutil
import numpy as np
import random
from skimage.io import imsave
from functools import reduce
from collections import OrderedDict


def import_config(config_name, prefix='configs'):
    cfg_path = '{}.{}'.format(prefix, config_name)
    m = importlib.import_module(name=cfg_path)
    os.makedirs(m.SNAPSHOT_DIR, exist_ok=True)
    shutil.copy(cfg_path.replace('.', '/') + '.py', os.path.join(m.SNAPSHOT_DIR, 'config.py'))
    return m


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter, cfg):
    if i_iter < cfg.PREHEAT_STEPS:
        lr = lr_warmup(cfg.LEARNING_RATE, i_iter, cfg.PREHEAT_STEPS)
    else:
        lr = lr_poly(cfg.LEARNING_RATE, i_iter, cfg.NUM_STEPS, cfg.POWER)
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr
    return lr


def get_console_file_logger(name, level=logging.INFO, logdir='./baseline'):
    logger = logging.Logger(name)
    logger.setLevel(level=level)
    logger.handlers = []
    BASIC_FORMAT = "%(asctime)s, %(levelname)s:%(name)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(level=level)

    fhlr = logging.FileHandler(os.path.join(logdir, str(time.time()) + '.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    return logger


def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True


def count_model_parameters(module, _default_logger=None):
    cnt = 0
    for p in module.parameters():
        cnt += reduce(lambda x, y: x * y, list(p.shape))
    _default_logger.info('#params: {}, {} M'.format(cnt, round(cnt / float(1e6), 3)))

    return cnt


if __name__ == '__main__':
    seed_torch(2333)
    s = torch.randn((5, 5)).cuda()
    print(s)