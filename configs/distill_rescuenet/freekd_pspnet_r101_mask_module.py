# _*_ coding : utf-8 _*_
# @Time: 2024/9/25 14:40
# @Author : Jianchong Guo
# @File : freekd_pspnet_r101_mask_module

from configs.base.rescuenet import TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER, IGNORE_LABEL, ITER_SIZE, PREHEAT_STEPS
import torch.nn as nn

TARGET_SET = 'RescueNet_FreeKD_PSPNet_r101_Mask_Module'

SNAPSHOT_DIR = './logs/rescuenet/freekd_pspnet_r101_mask_module_bs16_2k_8'

TEACHER_CKPT_PATH = './logs/rescuenet/rescuenet_pspnet_r101_bs16_60k.pth'

TOKENS = 8

TRAIN_DATA_CONFIG = TRAIN_DATA_CONFIG
TEST_DATA_CONFIG = TEST_DATA_CONFIG
WEIGHT_DECAY = WEIGHT_DECAY
MOMENTUM = MOMENTUM
LEARNING_RATE = 0.02
POWER = POWER
IGNORE_LABEL = IGNORE_LABEL
NUM_STEPS = 2000
NUM_STEPS_STOP = 2000
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
    classes=11,
    psp=dict(
        sizes=(1, 2, 3, 6),
        psp_size=2048,
        deep_features_size=1024
    )
)