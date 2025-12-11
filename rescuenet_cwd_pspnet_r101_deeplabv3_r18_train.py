# _*_ coding : utf-8 _*_
# @Time: 2024/9/14 16:37
# @Author : Jianchong Guo
# @File : rescuenet_cwd_pspnet_r101_deeplabv3_r18_train

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os.path as osp

from data.rescuenet import RescueNetLoader
from ever.core.iterator import Iterator
from tqdm import tqdm
from eval_rescuenet import evaluate
from module.baseline.pspnet import PSPNet
from module.baseline.deeplabv3 import DeepLabV3
from utils.tools import *
from losses.task import SegCrossEntropyLoss
from losses.cwd import CriterionCWD

parser = argparse.ArgumentParser(description='Run RescueNet_PSPNet_r101_DeepLabV3_r18 method.')

parser.add_argument('--config_path', type=str,
                    help='config path', default='distill_rescuenet.cwd_pspnet_r101_deeplabv3_r18')
args = parser.parse_args()
cfg = import_config(args.config_path)


def main():
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name=cfg.TARGET_SET, logdir=cfg.SNAPSHOT_DIR)

    teacher = PSPNet(cfg.TEACHER)

    teacher_ckpt = torch.load(cfg.TEACHER_CKPT_PATH)
    teacher.load_state_dict(teacher_ckpt, strict=True)
    logger.info('Teacher has load ckpt')

    student = DeepLabV3(cfg.STUDENT)

    criterion = SegCrossEntropyLoss(cfg.IGNORE_LABEL).cuda()
    criterion_cwd = CriterionCWD(512, 2048, norm_type='channel', divergence='kl', temperature=cfg.TEMPERATURE).cuda()

    teacher.eval()
    teacher.cuda()
    student.train()
    student.cuda()

    logger.info('exp = %s' % cfg.SNAPSHOT_DIR)

    logger.info('Teacher parameters:')
    count_model_parameters(teacher, logger)
    logger.info('Student parameters:')
    count_model_parameters(student, logger)

    train_loader = RescueNetLoader(cfg.TRAIN_DATA_CONFIG)
    train_loader_iter = Iterator(train_loader)

    epochs = cfg.NUM_STEPS_STOP / len(train_loader)
    logger.info('epochs ~= %.3f' % epochs)

    optimizer = optim.SGD([{'params': student.parameters()},
                           {'params': criterion_cwd.parameters()}], lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
                          weight_decay=cfg.WEIGHT_DECAY)

    optimizer.zero_grad()

    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter, cfg)

        for sub_i in range(cfg.ITER_SIZE):
            batch = train_loader_iter.next()
            images, labels = batch[0]
            images = images.cuda()
            labels['cls'] = labels['cls'].cuda()

            pred_student = student(images)
            with torch.no_grad():
                pred_teacher = teacher(images)

            task_loss = criterion(pred_student[-1], labels['cls'].long())
            cwd_loss_fea = cfg.ALPHA_CWD_FEA * criterion_cwd(pred_student[0], pred_teacher[0])
            cwd_loss_logit = cfg.ALPHA_CWD_LOGIT * criterion_cwd(pred_student[-1], pred_teacher[-1])

            loss = task_loss + cwd_loss_fea + cwd_loss_logit

            loss.backward()
            optimizer.step()

            if (i_iter + 1) % 50 == 0:
                logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
                logger.info(
                    'i_iter = %d, loss_total = %.6f, task_loss = %.6f, cwd_loss_fea = %.6f, cwd_loss_logit = %.6f lr = %.6f' % (
                        i_iter + 1, loss, task_loss, cwd_loss_fea, cwd_loss_logit, lr)
                )

            if i_iter >= cfg.NUM_STEPS_STOP - 1:
                print('save model ...')
                ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_' + str(cfg.NUM_STEPS_STOP) + '.pth')
                torch.save(student.state_dict(), ckpt_path)
                evaluate(student, cfg, ckpt_path=ckpt_path, logger=logger)
                break


if __name__ == '__main__':
    seed_torch(2333)
    main()