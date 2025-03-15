from backbone import BackBone
from corr import Corr
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamFC(nn.Module):

    def __init__(self):
        super(SiamFC, self).__init__()
        self.backbone = BackBone()
        self.corr = Corr()

    def forward(self, search, template):
        search_feature = self.backbone(search)
        template_feature = self.backbone(template)
        # print(search_feature.shape)
        # print(template_feature.shape)
        response_map = self.corr(search_feature, template_feature)
        return response_map


if __name__ == "__main__":
    search_tensor = torch.randn(1, 3, 255, 255)
    template_tensor = torch.randn(1, 3, 127, 127)
    siamfc = SiamFC()
    response_map = siamfc(search_tensor, template_tensor)
    # print(response_map.shape)
