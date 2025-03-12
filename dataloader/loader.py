import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import json


class SiamFCDataset(Dataset):

    def __init__(self, dataset_path, num_pairs=1, transform=None):
        self.dataset_path = dataset_path
        self.video_dir_list = os.listdir(self.dataset_path)
        self.transform = transform
        self.num_pairs = num_pairs

    def __len__(self):
        return len(self.video_dir_list)

    def __getitem__(self, idx):
        video_dir = self.video_dir_list[idx]
        imgs_list = os.path.join(self.dataset_path, video_dir, r"infrared")
        anno_path = os.path.join(self.dataset_path, video_dir, r"infrared.json")
        with open(anno_path, "r", encoding="utf-8") as file:
            anno = json.load(file)
        exist = anno["exist"]
        gt = anno["gt_rect"]
        length = len(exist)


if __name__ == "__main__":
    dataset = SiamFCDataset("data")
    a = dataset[0]
    print(1)
