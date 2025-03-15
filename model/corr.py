import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Corr(nn.Module):

    def __init__(self):
        super(Corr, self).__init__()

    def forward(self, x, z):  # x for search,z for template
        response_map = F.conv2d(x, z)
        return response_map


if __name__ == "__main__":
    x, z = torch.randn(1, 16, 32, 32), torch.randn(1, 16, 16, 16)
    corr = Corr()
    response_map = corr(x, z)
    print(torch.where(response_map == response_map.max()))
