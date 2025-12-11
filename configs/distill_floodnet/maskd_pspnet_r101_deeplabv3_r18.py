# _*_ coding : utf-8 _*_
# @Time: 2024/8/29 16:10
# @Author : Jianchong Guo
# @File : maskd_pspnet_r101_deeplabv3_r18

from configs.base.floodnet import TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER, IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS
import torch.nn as nn

TARGET_SET = 'FloodNet_MasKD_PSPNet_r101_DeepLabV3_r18'

SNAPSHOT_DIR = './logs/floodnet/maskd_pspnet_r101_deeplabv3_r18_bs16_60k'

TEACHER_CKPT_PATH = './logs/floodnet/floodnet_pspnet_r101_bs16_60k.pth'

MASK_CKPT = './logs/floodnet/maskd_pspnet_r101_mask_module_bs16_2k_8/FloodNet_MasKD_PSPNet_r101_Mask_Module_2000.pth'

ALPHA = 0.5
TOKENS = 8

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
            encoder_name='resnet18',
            decoder_channels=256,
            classes=10,
            encoder_weights='imagenet'
        )