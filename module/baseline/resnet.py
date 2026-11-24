"""Torchvision ResNet feature stages used by the PSPNet encoder."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision import models


class ResNetEncoder(nn.Module):
    def __init__(self, resnet_type='resnet50', include_conv5=True, batchnorm_trainable=True,
                 pretrained=False, freeze_at=0, output_stride=8, with_cp=(False,) * 4,
                 norm_layer=nn.BatchNorm2d, in_channels=3):
        super().__init__()
        if output_stride not in (8, 16, 32):
            raise ValueError("output_stride must be 8, 16 or 32")
        self.include_conv5 = include_conv5
        self.with_cp = with_cp
        self.batchnorm_trainable = batchnorm_trainable
        # Keep the backbone namespace used in the released experiment checkpoints.
        self.resnet = models.get_model(resnet_type, weights='IMAGENET1K_V1' if pretrained else None,
                                       norm_layer=norm_layer)
        del self.resnet.fc
        if in_channels != 3:
            conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                weights = self.resnet.conv1.weight.detach()
                channels = torch.arange(in_channels) % 3
                with torch.no_grad():
                    conv.weight.copy_(weights[:, channels] * (3 / in_channels))
            self.resnet.conv1 = conv

        for stride, layer in [(16, self.resnet.layer3), (32, self.resnet.layer4)]:
            if stride <= output_stride:
                continue
            for conv in layer.modules():
                if isinstance(conv, nn.Conv2d):
                    rate = stride // output_stride // conv.stride[0]
                    conv.stride = (1, 1)
                    if conv.kernel_size == (3, 3):
                        conv.padding = conv.dilation = (rate, rate)

        stages = [(self.resnet.conv1, self.resnet.bn1), (self.resnet.layer1,),
                  (self.resnet.layer2,), (self.resnet.layer3,), (self.resnet.layer4,)]
        for stage in stages[:max(0, freeze_at)]:
            for layer in stage:
                layer.requires_grad_(False)
        if not batchnorm_trainable:
            for layer in self.resnet.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.requires_grad_(False)
            self.train()

    def train(self, mode=True):
        super().train(mode)
        if mode and not self.batchnorm_trainable:
            for layer in self.resnet.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        return self

    def forward(self, x):
        backbone = self.resnet
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
        stages = [backbone.layer1, backbone.layer2, backbone.layer3]
        if self.include_conv5:
            stages.append(backbone.layer4)
        features = []
        for i, stage in enumerate(stages):
            if self.with_cp[i] and x.requires_grad:
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            features.append(x)
        return features
