"""Distillation losses for the published experiment configurations."""

from torch import nn
from pytorch_wavelets import DWTForward

from losses.cwd import CriterionCWD
from losses.freekd import FreeKDLoss
from losses.ifvd import CriterionIFV
from losses.kd import CriterionKD
from losses.maskd import MasKDLoss
from losses.mgd import CriterionMGD


class DistillationLoss(nn.Module):
    def __init__(self, method, cfg):
        super().__init__()
        self.method = method
        self.cfg = cfg
        classes = cfg.TEACHER['classes']
        if method in ('disasterkd', 'cwd'):
            self.cwd = CriterionCWD(512, 2048, norm_type='channel', divergence='kl', temperature=cfg.TEMPERATURE)
            if method == 'disasterkd':
                self.dwt = DWTForward(J=1, wave='haar')
        elif method == 'ifvd':
            self.ifvd = CriterionIFV(classes)
            self.kd = CriterionKD(cfg.TEMPERATURE)
        elif method == 'mgd':
            self.mgd = CriterionMGD(512, 2048, cfg.LAMBDA)
        else:
            criterion = {'maskd': MasKDLoss, 'freekd': FreeKDLoss}[method]
            self.mask_loss = criterion(channels=classes, num_tokens=cfg.TOKENS, pretrained=cfg.MASK_CKPT)
            self.mask_loss.requires_grad_(False)

    def forward(self, student, teacher, target, decoder):
        cfg = self.cfg
        if self.method == 'disasterkd':
            aligned = self.cwd.conv(student[0])
            cross_logits = decoder(aligned)
            low_s, high_s = self.dwt(cross_logits)
            low_t, high_t = self.dwt(teacher[-1])
            low_loss = self.cwd(low_s, low_t)
            high_loss = self.cwd(high_s[0].flatten(1, 2), high_t[0].flatten(1, 2))
            return cfg.ALPHA_LOGIT * (cfg.ALPHA_LF * low_loss + (1 - cfg.ALPHA_LF) * high_loss)
        if self.method == 'cwd':
            return (cfg.ALPHA_CWD_FEA * self.cwd(student[0], teacher[0])
                    + cfg.ALPHA_CWD_LOGIT * self.cwd(student[-1], teacher[-1]))
        if self.method == 'ifvd':
            return (cfg.ALPHA_IFVD * self.ifvd(student[0], teacher[0], target)
                    + cfg.ALPHA_KD * self.kd(student[-1], teacher[-1]))
        if self.method == 'mgd':
            return cfg.ALPHA_FEATURE * self.mgd(student[0], teacher[0])
        return cfg.ALPHA * self.mask_loss(student[-1], teacher[-1])
