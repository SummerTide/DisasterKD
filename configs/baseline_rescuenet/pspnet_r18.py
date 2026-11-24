# _*_ coding : utf-8 _*_
# @Time: 2024/9/13 1:58
# @Author : Jianchong Guo
# @File : pspnet_r18

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import pspnet

TARGET_SET = 'RescueNet_PSPNet_r18'

SNAPSHOT_DIR = './logs/rescuenet/pspnet_r18_bs16_60k'

MODEL = pspnet(18, classes=11)
