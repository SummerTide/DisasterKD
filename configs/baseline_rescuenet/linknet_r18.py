# _*_ coding : utf-8 _*_
# @Time: 2024/9/12 18:11
# @Author : Jianchong Guo
# @File : linknet_r18

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)

TARGET_SET = 'RescueNet_LinkNet_r18'

SNAPSHOT_DIR = './logs/rescuenet/linknet_r18_bs16_60k'

MODEL = dict(
            encoder_name='resnet18',
            classes=11,
            encoder_weights='imagenet'
        )
