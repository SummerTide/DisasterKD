# _*_ coding : utf-8 _*_
# @Time: 2024/8/30 10:21
# @Author : Jianchong Guo
# @File : ifvd_pspnet_r101_r18

from configs.base.floodnet import TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER, IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS
import torch.nn as nn

TARGET_SET = 'FloodNet_IFVD_PSPNet_r101_r18'

SNAPSHOT_DIR = './logs/floodnet/ifvd_pspnet_r101_r18_bs16_60k'

TEACHER_CKPT_PATH = './logs/floodnet/floodnet_pspnet_r101_bs16_60k.pth'

TEMPERATURE = 4.0

ALPHA_KD = 10.0
ALPHA_IFVD = 50.0

TRAIN_DATA_CONFIG = TRAIN_DATA_CONFIG
TEST_DATA_CONFIG = TEST_DATA_CONFIG
WEIGHT_DECAY = WEIGHT_DECAY
MOMENTUM = MOMENTUM
LEARNING_RATE = LEARNING_RATE
POWER = POWER
IGNORE_LABEL = IGNORE_LABEL
NUM_STEPS = NUM_STEPS
NUM_STEPS_STOP = NUM_STEPS_STOP
ITER_SIZE = ITER_SIZE
PREHEAT_STEPS = PREHEAT_STEPS

TEACHER = dict(
        encoder=dict(
            resnet_type='resnet101',
            include_conv5=True,
            batchnorm_trainable=True,
            pretrained=True,
            freeze_at=0,
            # 16 or 32
            output_stride=8,
            with_cp=(False, False, False, False),
            norm_layer=nn.BatchNorm2d,
        ),
        classes=10,
        psp=dict(
            sizes=(1, 2, 3, 6),
            psp_size=2048,
            deep_features_size=1024
        )
    )
    
STUDENT = dict(
        encoder=dict(
            resnet_type='resnet18',
            include_conv5=True,
            batchnorm_trainable=True,
            pretrained=True,
            freeze_at=0,
            # 16 or 32
            output_stride=8,
            with_cp=(False, False, False, False),
            norm_layer=nn.BatchNorm2d,
        ),
        classes=10,
        psp=dict(
            sizes=(1, 2, 3, 6),
            psp_size=512,
            deep_features_size=128
        )
    )