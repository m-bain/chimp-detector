#
#
# Author: Li Zhang <lz@robots.ox.ac.uk>
# Date  : 30 Sep. 2018
#

import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, c_mul=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96*c_mul, kernel_size=11, stride=2),
            nn.BatchNorm2d(96*c_mul),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96*c_mul, 256*c_mul, kernel_size=5),
            nn.BatchNorm2d(256*c_mul),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256*c_mul, 384*c_mul, kernel_size=3),
            nn.BatchNorm2d(384*c_mul),
            nn.ReLU(inplace=True),
            nn.Conv2d(384*c_mul, 384*c_mul, kernel_size=3),
            nn.BatchNorm2d(384*c_mul),
            nn.ReLU(inplace=True),
            nn.Conv2d(384*c_mul, 256*c_mul, kernel_size=3),
            nn.BatchNorm2d(256*c_mul),
        )
        self.feature_size = 256*c_mul

    def forward(self, x):
        x = self.features(x)
        return x

