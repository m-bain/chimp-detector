#
#
# Author: Li Zhang <lz@robots.ox.ac.uk>
# Date  : 30 Sep. 2018
#

import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


def conv2d_dw_group(x, kernel):
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3)) # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3)) # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class RPN_DW(RPN):
    def __init__(self, anchor_num=5, feature_in=256):
        super(RPN_DW, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in

        self.template_conv = nn.Conv2d(feature_in, feature_in, kernel_size=3)
        self.search_conv = nn.Conv2d(feature_in, feature_in, kernel_size=3)

        self.reg_conv = nn.Conv2d(feature_in, anchor_num*4, kernel_size=1)
        self.cls_conv = nn.Conv2d(feature_in, anchor_num*2, kernel_size=1)

    def forward(self, z_f, x_f):
        template = self.template_conv(z_f)
        search = self.search_conv(x_f)

        feature = conv2d_dw_group(search, template)

        reg = self.reg_conv(feature)
        cls = self.cls_conv(feature)
        return cls, reg
