#
#
# Author: Li Zhang <lz@robots.ox.ac.uk>
# Date  : 30 Sep. 2018
#

from svt.siamrpn import SiamRPN
from svt.features import AlexNet
from svt.rpn import RPN_DW


class Custom(SiamRPN):
    def __init__(self, **kwargs):
        super(Custom, self).__init__(**kwargs)
        self.features = AlexNet()
        feature_in = self.features.feature_size
        self.rpn_model = RPN_DW(anchor_num=self.anchor_num, feature_in=feature_in)

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc
