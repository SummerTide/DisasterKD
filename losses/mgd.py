# _*_ coding : utf-8 _*_
# @Time: 2024/8/30 11:09
# @Author : Jianchong Guo
# @File : mgd

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriterionMGD(nn.Module):
    def __init__(self, s_channels, t_channels, LAMBDA=0.75):
        super(CriterionMGD, self).__init__()

        self.feature_linear = torch.nn.Sequential(
            nn.Conv2d(s_channels, t_channels, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(t_channels),
            nn.ReLU(inplace=False)
        )

        self.generation = nn.Sequential(
            nn.Conv2d(t_channels, t_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(t_channels, t_channels, kernel_size=3, padding=1)
        )

        self.criterion = nn.MSELoss(reduction='sum')
        self.LAMBDA = LAMBDA

    def forward(self, pred_S, pred_T):
        n, c, h, w = pred_T.shape

        if pred_S.size(1) != pred_T.size(1):
            pred_S = self.feature_linear(pred_S)

        device = pred_S.device

        mat = torch.rand((n, 1, h, w)).to(device)
        mat = torch.where(mat > 1 - self.LAMBDA, 0, 1).to(device)

        masked = torch.mul(pred_S, mat)
        new_pred_S = self.generation(masked)
        loss = self.criterion(new_pred_S, pred_T) / n
        return loss
