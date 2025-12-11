# _*_ coding : utf-8 _*_
# @Time: 2024/9/24 15:17
# @Author : Jianchong Guo
# @File : testkd_pspnet_r101_deeplabv3_r18

from configs.base.rescuenet import TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER, \
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS
import torch.nn as nn

TARGET_SET = 'RescueNet_TestKD_PSPNet_r101_DeepLabV3_r18_para4'

SNAPSHOT_DIR = './logs/rescuenet/testkd_pspnet_r101_deeplabv3_r18_bs16_60k_para4'

TEACHER_CKPT_PATH = './logs/rescuenet/rescuenet_pspnet_r101_bs16_60k.pth'

TEMPERATURE = 4.0

ALPHA_FEA = 50.0
ALPHA_LOGIT = 3.0

ALPHA_LF = 0.4

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
    classes=11,
    psp=dict(
        sizes=(1, 2, 3, 6),
        psp_size=2048,
        deep_features_size=1024
    )
)

STUDENT = dict(
            encoder_name='resnet18',
            decoder_channels=256,
            classes=11,
            encoder_weights='imagenet'
        )