import torch
import torch.nn as nn
from torch.nn import functional as F

from model.BLSM import *
from model.FDGR import *


class BiMPan(nn.Module):
    def __init__(self, ms_channels=4, **kwargs):
        super(BiMPan, self).__init__()
        self.ms_channels = ms_channels
        self.BLSM = SDA_Block(4, 32)
        self.merging = nn.Sequential(
            nn.Conv2d(in_channels=ms_channels*32, out_channels=64,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
        )
        self.FDGR = nn.Sequential(
            nn.Conv2d(ms_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            FGMM(32),
            FGMM(32),
            FGMM(32),
            FGMM(32))
        self.fusion1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=self.ms_channels,
                      kernel_size=3, stride=1, padding=1),
        )

    def forward(self, pan, lms):
        f_list = []  # splitting bands
        for i in range(lms.shape[1]):
            p = pan.clone()
            s = lms[:, i, :, :].unsqueeze(dim=1)
            f = self.BLSM(p, s)
            f_list.append(f)
        f_cat = torch.cat(f_list, dim=1)
        local_f = self.merging(f_cat)
        pan_d = torch.cat([pan for _ in range(self.ms_channels)], dim=1)
        g = pan_d - lms
        global_f = self.FDGR(g)
        f_fused = torch.cat([global_f, local_f], dim=1)
        f_res = self.fusion1(f_fused)
        res = self.fusion2(f_res)
        return res + lms


if __name__ == '__main__':
    from torchsummary import summary
    N = BiMPan(ms_channels=8)
    # summary(N, [(1,64,64), (8,64,64)], device='cpu')
    pan = torch.rand(1, 1, 64, 64)
    lms = torch.rand(1, 8, 64, 64)
    result = N(pan, lms)
    print(result.shape)
