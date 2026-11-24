# _*_ coding : utf-8 _*_
# @Time: 2024/9/2 10:44
# @Author : Jianchong Guo
# @File : mgd_pspnet_r101_deeplabv3_r18

from configs.base.floodnet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)
from configs.models import deeplabv3, pspnet

TARGET_SET = 'FloodNet_MGD_PSPNet_r101_DeepLabV3_r18'

SNAPSHOT_DIR = './logs/floodnet/mgd_pspnet_r101_deeplabv3_r18_bs16_60k'

TEACHER_CKPT_PATH = './logs/floodnet/floodnet_pspnet_r101_bs16_60k.pth'

ALPHA_FEATURE = 0.00002
LAMBDA = 0.75

TEACHER = pspnet(101, classes=10)
    
STUDENT = deeplabv3(18, classes=10)
