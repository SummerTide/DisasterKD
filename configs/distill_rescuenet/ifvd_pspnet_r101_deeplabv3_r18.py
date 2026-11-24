# _*_ coding : utf-8 _*_
# @Time: 2024/8/30 10:21
# @Author : Jianchong Guo
# @File : ifvd_deeplabv3_r101_pspnet_r18

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import deeplabv3, pspnet

TARGET_SET = 'RescueNet_IFVD_PSPNet_r101_DeepLabV3_r18'

SNAPSHOT_DIR = './logs/rescuenet/ifvd_pspnet_r101_deeplabv3_r18_bs16_60k'

TEACHER_CKPT_PATH = './logs/rescuenet/rescuenet_pspnet_r101_bs16_60k.pth'

TEMPERATURE = 4.0

ALPHA_IFVD = 50.0

TEACHER = pspnet(101, classes=11)

STUDENT = deeplabv3(18, classes=11)
