import argparse

import torch

from module.baseline.deeplabv3 import DeepLabV3
from module.baseline.pspnet import PSPNet
from utils.tools import get_console_file_logger, import_config, seed_torch


def run_evaluation(evaluate, default_config, default_checkpoint):
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint")
    parser.add_argument('--config_path', default=default_config)
    parser.add_argument('--checkpoint', default=default_checkpoint)
    args = parser.parse_args()
    seed_torch(2333)
    cfg = import_config(args.config_path)
    options = dict(cfg.MODEL if hasattr(cfg, 'MODEL') else cfg.STUDENT)
    if 'encoder' in options:
        options['encoder'] = dict(options['encoder'], pretrained=False)
        model = PSPNet(options)
    else:
        options['encoder_weights'] = None
        model = DeepLabV3(options)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_console_file_logger(cfg.TARGET_SET, logdir=cfg.SNAPSHOT_DIR)
    evaluate(model, cfg, ckpt_path=args.checkpoint, logger=logger)
