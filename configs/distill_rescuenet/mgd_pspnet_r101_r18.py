# _*_ coding : utf-8 _*_
# @Time: 2024/8/30 11:05
# @Author : Jianchong Guo
# @File : mgd_pspnet_r101_r18

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import pspnet

TARGET_SET = 'RescueNet_MGD_PSPNet_r101_r18'

SNAPSHOT_DIR = './logs/rescuenet/mgd_pspnet_r101_r18_bs16_60k'

TEACHER_CKPT_PATH = './logs/rescuenet/rescuenet_pspnet_r101_bs16_60k.pth'

ALPHA_FEATURE = 0.00002
LAMBDA = 0.75

TEACHER = pspnet(101, classes=11)
    
STUDENT = pspnet(18, classes=11)
