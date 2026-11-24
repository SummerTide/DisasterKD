# _*_ coding : utf-8 _*_
# @Time: 2024/8/4 9:50
# @Author : Jianchong Guo
# @File : deeplabv3_r101

from configs.base.floodnet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import pspnet

TARGET_SET = 'FloodNet_PSPNet_r101'

SNAPSHOT_DIR = './logs/floodnet/pspnet_r101_bs16_60k'

MODEL = pspnet(101, classes=10)
