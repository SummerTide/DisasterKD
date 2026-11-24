# _*_ coding : utf-8 _*_
# @Time: 2024/9/24 10:01
# @Author : Jianchong Guo
# @File : testkd_pspnet_r101_deeplabv3_r18

from configs.base.floodnet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import deeplabv3, pspnet

TARGET_SET = 'FloodNet_TestKD_PSPNet_r101_DeepLabV3_r18_wofea_para4'

SNAPSHOT_DIR = './logs/floodnet/testkd_pspnet_r101_deeplabv3_r18_bs16_60k_wofea_para4'

TEACHER_CKPT_PATH = './logs/floodnet/floodnet_pspnet_r101_bs16_60k.pth'

TEMPERATURE = 4.0

ALPHA_FEA = 50.0
ALPHA_LOGIT = 3.0

ALPHA_LF = 0.4

TEACHER = pspnet(101, classes=10)

STUDENT = deeplabv3(18, classes=10)
