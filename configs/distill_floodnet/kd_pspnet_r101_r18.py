# _*_ coding : utf-8 _*_
# @Time: 2024/9/19 14:44
# @Author : Jianchong Guo
# @File : kd_pspnet_r101_r18

from configs.base.floodnet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import pspnet

TARGET_SET = 'FloodNet_CrossKD_PSPNet_r101_r18'

SNAPSHOT_DIR = './logs/floodnet/crosskd_pspnet_r101_r18_bs16_60k'

TEACHER_CKPT_PATH = './logs/floodnet/floodnet_pspnet_r101_bs16_60k.pth'

TEMPERATURE = 4.0
ALPHA_KD = 1.0

TEACHER = pspnet(101, classes=10)

STUDENT = pspnet(18, classes=10)
