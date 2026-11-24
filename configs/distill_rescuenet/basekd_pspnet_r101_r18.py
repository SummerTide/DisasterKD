# _*_ coding : utf-8 _*_
# @Time: 2024/9/18 22:16
# @Author : Jianchong Guo
# @File : basekd_pspnet_r101_r18

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import pspnet

TARGET_SET = 'RescueNet_CrossCWD_PSPNet_r101_r18'

SNAPSHOT_DIR = './logs/rescuenet/crosscwd_pspnet_r101_r18_bs16_60k'

TEACHER_CKPT_PATH = './logs/rescuenet/rescuenet_pspnet_r101_bs16_60k.pth'

TEMPERATURE = 4.0

ALPHA_CWD_LOGIT = 3.0
ALPHA_CWD_FEA = 50.0

TEACHER = pspnet(101, classes=11)

STUDENT = pspnet(18, classes=11)
