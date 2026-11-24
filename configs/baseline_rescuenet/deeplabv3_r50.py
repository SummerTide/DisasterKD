# _*_ coding : utf-8 _*_
# @Time: 2024/9/13 12:21
# @Author : Jianchong Guo
# @File : deeplabv3_r50

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import deeplabv3

TARGET_SET = 'RescueNet_DeepLabV3_r50'

SNAPSHOT_DIR = './logs/rescuenet/deeplabv3_r50_bs16_60k'

MODEL = deeplabv3(50, classes=11)
