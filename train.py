import torch
import torch.nn as nn
import model
import torch
from dataloader import SiamFCDataset
from dataloader.transformer import Transformer
from utils import _BalancedLoss, _collate_fn, _create_labels
from torch.utils.data import DataLoader
import torch.optim as optim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SiamFC = model.SiamFC().to(device)
    optimizer = optim.Adam(SiamFC.parameters(), lr=1e-3)
    dataset = SiamFCDataset("data", 10, Transformer())
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=_collate_fn)
    loss_func = _BalancedLoss(neg_weight=0.5)
    for epoch in range(500):
        total_loss = 0
        for template, search in dataloader:
            optimizer.zero_grad()
            template = template.to(device)
            search = search.to(device)
            response_map = SiamFC.forward_backbone(template, search)
            labels = _create_labels(response_map.shape).to(device)
            loss = loss_func(response_map, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print("Epoch:{}\t\tLoss:{}".format(epoch, total_loss))
    torch.save(SiamFC, 'siamfc_full.pth')


if __name__ == "__main__":
    main()
