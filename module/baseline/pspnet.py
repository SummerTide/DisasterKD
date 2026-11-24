# _*_ coding : utf-8 _*_
# @Time: 2024/7/22 14:53
# @Author : Jianchong Guo
# @File : pspnet

import torch
from torch import nn
from torch.nn import functional as F
from module.baseline.resnet import ResNetEncoder


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feats = ResNetEncoder(**config.get('encoder', {}))
        psp = config.get('psp', {})
        channels = psp.get('deep_features_size', 1024)
        self.psp = PSPModule(psp.get('psp_size', 2048), channels, psp.get('sizes', (1, 2, 3, 6)))
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(channels, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Conv2d(64, config.get('classes', 10), kernel_size=1)

    def forward(self, x):
        '''
        for example: PSPNet_r101    vs  PSPNet_r18
        psp=dict(
            sizes=(1, 2, 3, 6),
            psp_size=2048,
            deep_features_size=1024
        )
        psp=dict(
            sizes=(1, 2, 3, 6),
            psp_size=512,
            deep_features_size=128
        )
        :param x:   Tensor:(1, 3, 512, 512) n, c, h, w
        '''
        f = self.feats(x)[-1]   # Tensor:(1, 2048, 64, 64)      (1, 512, 64, 64)
        p = self.psp(f)
        psp = p
        p = self.drop_1(p)  # Tensor:(1, 1024, 64, 64)  (1, 128, 64, 64)

        p = self.up_1(p)
        p = self.drop_2(p)  # Tensor: (1, 256, 128, 128)    (1, 256, 128, 128)

        p = self.up_2(p)
        p = self.drop_2(p)  # Tensor:(1, 64, 256, 256)  (1, 64, 256, 256)

        p = self.up_3(p)
        p = self.drop_2(p)  # Tensor:(1, 64, 512, 512)  (1, 64, 512, 512)
        logit = self.final(p)   # Tensor:(1, 10, 512, 512)  (1, 10, 512, 512)

        return [f, psp, logit]

    def forward_feature(self, f):
        '''
        Args:
            for r101
            f: (N, 2048, 64, 64)

        Returns:
            logit
        '''
        p = self.psp(f)
        p = self.drop_1(p)  # Tensor:(1, 1024, 64, 64)  (1, 128, 64, 64)

        p = self.up_1(p)
        p = self.drop_2(p)  # Tensor: (1, 256, 128, 128)    (1, 256, 128, 128)

        p = self.up_2(p)
        p = self.drop_2(p)  # Tensor:(1, 64, 256, 256)  (1, 64, 256, 256)

        p = self.up_3(p)
        p = self.drop_2(p)  # Tensor:(1, 64, 512, 512)  (1, 64, 512, 512)
        logit = self.final(p)  # Tensor:(1, 10, 512, 512)  (1, 10, 512, 512)
        return logit

if __name__ == '__main__':
    m = PSPNet(dict())
    m.eval()
    o = m(torch.ones(2, 3, 512, 512))
    print(o[-1].shape)