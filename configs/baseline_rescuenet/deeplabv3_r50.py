# _*_ coding : utf-8 _*_
# @Time: 2024/9/13 12:21
# @Author : Jianchong Guo
# @File : deeplabv3_r50

from configs.base.rescuenet import TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER, IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS

TARGET_SET = 'RescueNet_DeepLabV3_r50'

SNAPSHOT_DIR = './logs/rescuenet/deeplabv3_r50_bs16_60k'

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

MODEL = dict(
            encoder_name='resnet50',
            decoder_channels=256,
            classes=11,
            encoder_weights='imagenet'
        )