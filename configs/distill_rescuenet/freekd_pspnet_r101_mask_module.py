# _*_ coding : utf-8 _*_
# @Time: 2024/9/25 14:40
# @Author : Jianchong Guo
# @File : freekd_pspnet_r101_mask_module

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import pspnet

TARGET_SET = 'RescueNet_FreeKD_PSPNet_r101_Mask_Module'

SNAPSHOT_DIR = './logs/rescuenet/freekd_pspnet_r101_mask_module_bs16_2k_8'

TEACHER_CKPT_PATH = './logs/rescuenet/rescuenet_pspnet_r101_bs16_60k.pth'

TOKENS = 8

LEARNING_RATE = 0.02
NUM_STEPS = 2000
NUM_STEPS_STOP = 2000

TEACHER = pspnet(101, classes=11)
