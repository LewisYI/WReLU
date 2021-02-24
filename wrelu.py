import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

class WReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, x + self.bn(self.conv(x)))


class WReLU_light(nn.Module):
    def __init__(self, in_channels):  # ch_in, kernel
        super().__init__()
        self.conv_frelu1 = nn.Conv2d(in_channels, in_channels, (1,3), 1, (0,1), groups=in_channels, bias=False)
        self.conv_frelu2 = nn.Conv2d(in_channels, in_channels, (3,1), 1, (1,0), groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu1(x)
        x1 = self.bn1(x1)
        x2 = self.conv_frelu2(x)
        x2 = self.bn2(x2)
        x = torch.max(x, x+x1+x2)
        return x
