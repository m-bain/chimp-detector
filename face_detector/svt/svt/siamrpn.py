#
#
# Author: Li Zhang <lz@robots.ox.ac.uk>
# Date  : 30 Sep. 2018
#

import torch.nn as nn
import torch.nn.functional as F


class SiamRPN(nn.Module):
    def __init__(self, anchors=None):
        super(SiamRPN, self).__init__()
        self.anchors = anchors
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.features = None
        self.rpn_model = None

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def forward(self, template, search):
        template_feature = self.feature_extractor(template)
        search_feature = self.feature_extractor(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)

        rpn_pred_cls = rpn_pred_cls.permute(0, 2, 3, 1).contiguous()
        rpn_pred_cls = F.softmax(rpn_pred_cls.view(-1, 2), dim=1).view_as(rpn_pred_cls)
        rpn_pred_cls = rpn_pred_cls.permute(0, 3, 1, 2)

        return rpn_pred_loc, rpn_pred_cls

    def temple(self, z):
        self.zf = self.feature_extractor(z)

    def track(self, x):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, xf)
        return rpn_pred_cls, rpn_pred_loc

