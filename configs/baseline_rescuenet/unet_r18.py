# _*_ coding : utf-8 _*_
# @Time: 2024/9/3 13:20
# @Author : Jianchong Guo
# @File : unet_r18

from configs.base.rescuenet import (
    TRAIN_DATA_CONFIG, TEST_DATA_CONFIG, WEIGHT_DECAY, MOMENTUM, LEARNING_RATE, POWER,
    IGNORE_LABEL, NUM_STEPS, NUM_STEPS_STOP, ITER_SIZE, PREHEAT_STEPS,
)

TARGET_SET = 'RescueNet_UNet_r18'

SNAPSHOT_DIR = './logs/rescuenet/unet_r18_bs16_60k'
