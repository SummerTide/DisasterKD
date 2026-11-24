# _*_ coding : utf-8 _*_
# @Time: 2024/9/25 15:25
# @Author : Jianchong Guo
# @File : freekd_pspnet_r101_r18

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import pspnet

TARGET_SET = 'RescueNet_FreeKD_PSPNet_r101_r18_30'

SNAPSHOT_DIR = './logs/rescuenet/freekd_pspnet_r101_r18_bs16_60k_30'

TEACHER_CKPT_PATH = './logs/rescuenet/rescuenet_pspnet_r101_bs16_60k.pth'

MASK_CKPT = './logs/rescuenet/freekd_pspnet_r101_mask_module_bs16_2k_8/RescueNet_FreeKD_PSPNet_r101_Mask_Module_2000.pth'

ALPHA = 30
TOKENS = 8

TEACHER = pspnet(101, classes=11)

STUDENT = pspnet(18, classes=11)
