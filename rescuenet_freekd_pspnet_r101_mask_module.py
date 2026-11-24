# _*_ coding : utf-8 _*_
# @Time: 2024/9/25 14:41
# @Author : Jianchong Guo
# @File : rescuenet_freekd_pspnet_r101_mask_module

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os.path as osp

from data.rescuenet import RescueNetLoader
from data.loader import repeat_batches
from tqdm import tqdm
from module.baseline.pspnet import PSPNet
from utils.tools import *
import torch.nn.functional as F
from losses.freekd import MaskModule
from losses.task import SegCrossEntropyLoss

parser = argparse.ArgumentParser(description='Run RescueNet_FreeKD_PSPNet_r101_Mask_Module method.')

parser.add_argument('--config_path', type=str,
                    help='config path', default='distill_rescuenet.freekd_pspnet_r101_mask_module')
args = parser.parse_args()
cfg = import_config(args.config_path)


def main():
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name=cfg.TARGET_SET, logdir=cfg.SNAPSHOT_DIR)

    teacher = PSPNet(cfg.TEACHER)

    teacher_ckpt = torch.load(cfg.TEACHER_CKPT_PATH)
    teacher.load_state_dict(teacher_ckpt, strict=True)
    logger.info('Teacher has load ckpt')

    teacher.train()
    teacher.cuda()

    logger.info('exp = %s' % cfg.SNAPSHOT_DIR)

    logger.info('Teacher parameters:')
    count_model_parameters(teacher, logger)

    train_loader = RescueNetLoader(cfg.TRAIN_DATA_CONFIG)
    train_loader_iter = repeat_batches(train_loader)

    epochs = cfg.NUM_STEPS_STOP / len(train_loader)
    logger.info('epochs ~= %.3f' % epochs)

    mask_module = MaskModule(channels=11, num_tokens=cfg.TOKENS).cuda()
    criterion = SegCrossEntropyLoss().cuda()

    optimizer = optim.SGD([{'params': mask_module.parameters()}], lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
                          weight_decay=cfg.WEIGHT_DECAY)

    optimizer.zero_grad()

    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter, cfg)

        for sub_i in range(cfg.ITER_SIZE):
            images, labels = next(train_loader_iter)
            images = images.cuda()
            labels['cls'] = labels['cls'].cuda()

            with torch.no_grad():
                pred_teacher = teacher(images)

            pred_teacher[-1], mask_loss = mask_module(pred_teacher[-1])

            task_loss = criterion(pred_teacher[-1], labels['cls'].long())

            loss = task_loss + mask_loss

            loss.backward()
            optimizer.step()

            if (i_iter + 1) % 50 == 0:
                logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
                logger.info(
                    'i_iter = %d, loss_total = %.6f, task_loss = %.6f, mask_loss = %.6f, lr = %.6f' % (
                        i_iter + 1, loss, task_loss, mask_loss, lr)
                )

            if i_iter >= cfg.NUM_STEPS_STOP - 1:
                print('save model ...')
                ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_' + str(cfg.NUM_STEPS_STOP) + '.pth')
                torch.save(mask_module.state_dict(), ckpt_path)
                break


if __name__ == '__main__':
    seed_torch(2333)
    main()
