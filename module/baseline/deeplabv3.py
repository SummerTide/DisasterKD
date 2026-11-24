# _*_ coding : utf-8 _*_
# @Time: 2024/8/12 9:45
# @Author : Jianchong Guo
# @File : deeplabv3
# @Project : pspnet.py

import segmentation_models_pytorch as smp
from torch import nn


class DeepLabV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = smp.DeepLabV3(
            encoder_name=config.get('encoder_name', 'resnet18'),
            encoder_weights=config.get('encoder_weights'),
            decoder_channels=config.get('decoder_channels', 256),
            classes=config.get('classes', 10),
            activation=None,
        )

    def forward(self, x):
        f = self.features.encoder(x)
        p = self.features.decoder(*f)
        logit = self.features.segmentation_head(p)

        return [f[-1], p, logit]

    def forward_feature(self, f):
        f = [f]
        p = self.features.decoder(*f)
        logit = self.features.segmentation_head(p)
        return logit

if __name__ == '__main__':
    import torch
    m = DeepLabV3(dict())
    m.eval()
    o = m(torch.ones(2, 3, 512, 512))
    print(o[-1].shape)