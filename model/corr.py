import torch
import torch.nn as nn
import torch.nn.functional as F


class Corr(nn.Module):

    def __init__(self):
        super(Corr, self).__init__()

    def forward(self, x, z):  # x for search,z for template
        """
        TODO: More efficient implementation
        """
        B, C, kz, _ = z.shape
        _, _, kx, _ = x.shape
        out = []
        for i in range(B):
            res = F.conv2d(z[i:i + 1], x[i:i + 1])  # [1, 1, h, w]
            out.append(res)
        return torch.cat(out, dim=0)


if __name__ == "__main__":
    x, z = torch.randn(10, 16, 32, 32), torch.randn(10, 16, 16, 16)
    corr = Corr()
    response_map = corr(x, z)
    # print(torch.where(response_map == response_map.max()))
    print(response_map.shape)
