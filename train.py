import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import _collate_fn, _BCEWeights
import model
from dataloader import SiamFCDataset
import torch.optim as optim
import torch
import numpy as np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SiamFC = model.SiamFC().to(device)
    optimizer = optim.Adam(SiamFC.parameters(), lr=1e-3)
    dataset = SiamFCDataset("data", 1)
    loss_func = _BCEWeights()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn)
    for epoch in range(500):
        total_loss = 0
        for search, template, labels, centers in dataloader:
            search, template, labels = search.to(device), template.to(device), labels.to(device)
            response_map = SiamFC.forward(search, template)
            optimizer.zero_grad()
            loss = loss_func(response_map, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print("Epoch:{}\tLoss:{}".format(epoch, total_loss))
    torch.save(SiamFC, 'siamfc_full.pth')


if __name__ == "__main__":
    main()
