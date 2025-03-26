from .backbone import BackBone
from .corr import Corr
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamFC(nn.Module):

    def __init__(self):
        super(SiamFC, self).__init__()
        self.backbone = BackBone()
        self.corr = Corr()

    def forward_backbone(self, template, search):
        search_feature = self.backbone(search)
        template_feature = self.backbone(template)
        # print(search_feature.shape)
        # print(template_feature.shape)
        response_map = self.corr(search_feature, template_feature)
        return response_map

    def forward(self, search, template):
        return self.forward_backbone(search, template)

    def head(self, response_map):
        pass
