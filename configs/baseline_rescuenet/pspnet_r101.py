# _*_ coding : utf-8 _*_
# @Time: 2024/9/1 11:00
# @Author : Jianchong Guo
# @File : pspnet_r101

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import pspnet

TARGET_SET = 'RescueNet_PSPNet_r101'

SNAPSHOT_DIR = './logs/rescuenet/pspnet_r101_bs16_60k'

MODEL = pspnet(101, classes=11)
