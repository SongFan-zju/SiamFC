import torch
import numpy as np
import torchvision
import torch.nn as nn


class BackBone(nn.Module):

    def __init__(self):
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 8, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(8, 16, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        return x


if __name__ == "__main__":
    search_tensor = torch.randn(1, 3, 255, 255)
    template_tensor = torch.randn(1, 3, 127, 127)
    backbone = BackBone()
    search_feature = backbone(search_tensor)
    template_feature = backbone(template_tensor)
    print(search_feature.shape)
    print(template_feature.shape)
