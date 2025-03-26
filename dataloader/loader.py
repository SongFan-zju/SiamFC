import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from .transformer import Transformer


class SiamFCDataset(Dataset):

    def __init__(self, dataset_path, num_pairs=10, transform=None):
        super(SiamFCDataset, self).__init__()
        self.dataset_path = dataset_path
        self.video_dir_list = os.listdir(self.dataset_path)
        self.num_pairs = num_pairs
        self.transform = transform

    def __len__(self):
        return len(self.video_dir_list)

    def _filter(self, anno):

        def _bbox_filter(bbox):
            x, y, w, h = bbox
            if x < 200 or x > 350: return False
            if y < 250 or y > 300: return False
            if w * h < 20: return False
            if w < 20 or h < 20: return False
            if w >= 100 or h >= 100: return False
            if w / h < 0.25 or w / h > 4: return False
            return True

        exist = np.array(anno["exist"])
        bboxs = anno["gt_rect"]
        is_exist = np.where(exist == 1)[0]
        val_indice = []
        for i in is_exist:
            if _bbox_filter(bboxs[i]):
                val_indice.append(i)
        return np.array(val_indice)

    def _sample_pair(self, indices):
        n = len(indices)
        pairs = []
        max_try = 100
        while len(pairs) < self.num_pairs:
            rand_z, rand_x = np.sort(np.random.choice(indices, 2))
            if rand_x - rand_z <= 20:
                pairs.append([rand_z, rand_x])
        return pairs

    def __getitem__(self, index):
        data_dir = os.path.join(self.dataset_path, self.video_dir_list[index])
        imgs_dir = os.path.join(data_dir, "infrared")
        imgs_list = os.listdir(imgs_dir)
        anno_path = os.path.join(data_dir, "infrared.json")
        with open(anno_path, "r") as f:
            anno = json.load(f)
        val_indice = self._filter(anno)
        pairs = self._sample_pair(val_indice)
        img_z_list, img_x_list = [], []
        for z, x in pairs:
            img_z = cv2.imread(os.path.join(imgs_dir, imgs_list[z]), cv2.IMREAD_COLOR)
            img_x = cv2.imread(os.path.join(imgs_dir, imgs_list[x]), cv2.IMREAD_COLOR)
            img_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB)
            img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
            # img_z[0:85, :, :] = 100
            # img_x[0:85, :, :] = 100
            # cv2.imshow('original', img_z)
            # cv2.waitKey(0)
            bbox_z = anno["gt_rect"][z]
            bbox_x = anno["gt_rect"][x]
            if self.transform:
                img_z, img_x = self.transform(img_z, img_x, bbox_z, bbox_x)
                # print(img_z.shape, img_x.shape)
                img_z_list.append(img_z)
                img_x_list.append(img_x)
            else:
                img_z_list.append(img_z)
                img_x_list.append(img_x)
        img_z_list = torch.stack(img_z_list, dim=0)
        img_x_list = torch.stack(img_x_list, dim=0)
        return img_z_list, img_x_list


if __name__ == "__main__":
    dataset = SiamFCDataset("data", 10, Transformer())
    a = dataset[0]
    print(a)
