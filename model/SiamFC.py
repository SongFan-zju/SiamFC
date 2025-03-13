import torch
import numpy as np
import torchvision
import torch.nn as nn


class BackBone(nn.Module):

    def __init__(self):
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
