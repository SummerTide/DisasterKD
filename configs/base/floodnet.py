# _*_ coding : utf-8 _*_
# @Time: 2024/8/4 9:46
# @Author : Jianchong Guo
# @File : floodnet
# @Project : Knowledge_Distillation_UAV

from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop, RandomBrightnessContrast
from albumentations import Compose, OneOf, Normalize
from albumentations.pytorch import ToTensorV2
from data.transforms import DivisiblePad, RandomDiscreteScale

from data.paths import split_dirs

train_dir = split_dirs('floodnet', 'train')
test_dir = split_dirs('floodnet', 'test')

TRAIN_DATA_CONFIG = dict(
    image_dir=train_dir['image_dir'],
    mask_dir=train_dir['mask_dir'],
    transforms=Compose([
        RandomDiscreteScale([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
        RandomCrop(512, 512),
        RandomBrightnessContrast(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        ToTensorV2()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=4,
)

TEST_DATA_CONFIG = dict(
    image_dir=test_dir['image_dir'],
    mask_dir=test_dir['mask_dir'],
    transforms=Compose([
        DivisiblePad(size_divisor=32, always_apply=True),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        ToTensorV2()
    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=0,
)

# Hyper Parameters
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9

LEARNING_RATE = 0.01
POWER = 0.9

IGNORE_LABEL = -1

NUM_STEPS = 120000
NUM_STEPS_STOP = 120000  # Use damping instead of early stopping

ITER_SIZE = 1
PREHEAT_STEPS = 0  # no preheat
