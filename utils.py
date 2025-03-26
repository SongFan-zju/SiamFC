import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _create_labels(size=15):

    def logistic_labels(x, y, r_pos):
        # x^2+y^2<4 的位置设为为1，其他为0
        dist = np.sqrt(x**2 + y**2)
        labels = np.where(
            dist <= r_pos,  #r_os=2
            np.ones_like(x),  #np.ones_like(x),用1填充x
            np.zeros_like(x))  #np.zeros_like(x),用0填充x
        return labels

    # distances along x- and y-axis
    n, c, h, w = size  # [8,1,15,15]
    x = np.arange(w) - (w - 1) / 2  #x=[-7 -6 ....0....6 7]
    y = np.arange(h) - (h - 1) / 2  #y=[-7 -6 ....0....6 7]
    x, y = np.meshgrid(x, y)
    #建立标签
    r_pos = 2  # 16/8
    labels = logistic_labels(x, y, r_pos)
    #重复batch_size个label，因为网络输出是batch_size张response map
    labels = labels.reshape((1, 1, h, w))  #[1,1,15,15]
    labels = np.tile(labels, (n, c, 1, 1))  #将labels扩展[8,1,15,15]
    return torch.from_numpy(labels)


class _BalancedLoss(nn.Module):

    def __init__(self, neg_weight=0.5):
        super(_BalancedLoss, self).__init__()
        self.neg_weight = neg_weight

    def forward(self, input, target):  #属于BalancedLoss的forward
        pos_mask = (target == 1)  #相应位置标注为True（1）
        neg_mask = (target == 0)  #相应位置标注为True（0）
        pos_num = pos_mask.sum().float()  #计算True的个数（1）
        neg_num = neg_mask.sum().float()  #计算True的个数（0）
        weight = target.new_zeros(target.size())  #创建一个大小与target相同的weight
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        #binary_cross_entropy_with_logits等价于sigmod+F.binary_cross_entropy！！！
        return F.binary_cross_entropy_with_logits(  #torch.nn.functional.binary_cross_entropy_with_logits
            input, target, weight, reduction='sum')


def _collate_fn(batch):
    # batch 是一个 list，里面是 [(s1, t1, l1), ..., (s10, t10, l10)] × B
    search_list, template_list = [], []
    for sample in batch:
        template_list.append(sample[0])
        search_list.append(sample[1])
    return (
        torch.cat(search_list, dim=0),  # [B*p, 3, 255, 255]
        torch.cat(template_list, dim=0),  # [B*p, 3, 127, 127]
    )


if __name__ == "__main__":
    response_size = 17
    label = _create_labels(response_size)
    print(label.shape)
    print(label[8, 1])
