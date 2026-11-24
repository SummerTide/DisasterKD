# _*_ coding : utf-8 _*_
# @Time: 2024/9/22 13:40
# @Author : Jianchong Guo
# @File : testkd_pspnet_r101_r18

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import pspnet

TARGET_SET = 'RescueNet_TestKD_PSPNet_r101_r18_wofea_para0'

SNAPSHOT_DIR = './logs/rescuenet/testkd_pspnet_r101_r18_bs16_60k_wofea_para0'

TEACHER_CKPT_PATH = './logs/rescuenet/rescuenet_pspnet_r101_bs16_60k.pth'

TEMPERATURE = 4.0

ALPHA_FEA = 50.0
ALPHA_LOGIT = 3.0

ALPHA_LF = 0.0

TEACHER = pspnet(101, classes=11)

STUDENT = pspnet(18, classes=11)
