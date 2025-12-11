# _*_ coding : utf-8 _*_
# @Time: 2024/8/12 9:45
# @Author : Jianchong Guo
# @File : deeplabv3
# @Project : pspnet.py

import segmentation_models_pytorch as smp
import ever as er


@er.registry.MODEL.register()
class DeepLabV3(er.ERModule, ):
    def __init__(self, config):
        super(DeepLabV3, self).__init__(config)
        self.features = smp.DeepLabV3(self.config.encoder_name,
                                    encoder_weights=self.config.encoder_weights,
                                    classes=self.config.classes,
                                    activation=None
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

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet18',
            classes=10,
            encoder_weights=None,
        ))


if __name__ == '__main__':
    import torch
    m = DeepLabV3(dict())
    m.eval()
    o = m(torch.ones(2, 3, 512, 512))
    print(o.shape)