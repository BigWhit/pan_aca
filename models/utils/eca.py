import math

import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict


class ECAAttention(nn.Module):

    def __init__(self,channels, gamma=2, b=0):  # gamma=2,b=1
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        if (t % 2):
            k_size = t
        else:
            k_size = t + 1
        # self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)

if __name__ == '__main__':
    input = torch.randn(8, 64, 184, 184)
    eca = ECAAttention(128)
    output = eca(input)
    print(output.shape)
