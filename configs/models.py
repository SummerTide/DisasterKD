"""Model definitions shared by the FloodNet and RescueNet experiment configs."""

from torch import nn


def pspnet(depth, classes):
    psp_size, decoder_channels = {18: (512, 128), 101: (2048, 1024)}[depth]
    return dict(
        encoder=dict(resnet_type=f'resnet{depth}', include_conv5=True,
                     batchnorm_trainable=True, pretrained=True, freeze_at=0,
                     output_stride=8, with_cp=(False,) * 4, norm_layer=nn.BatchNorm2d),
        classes=classes,
        psp=dict(sizes=(1, 2, 3, 6), psp_size=psp_size, deep_features_size=decoder_channels),
    )


def deeplabv3(depth, classes):
    return dict(encoder_name=f'resnet{depth}', decoder_channels=256,
                classes=classes, encoder_weights='imagenet')
