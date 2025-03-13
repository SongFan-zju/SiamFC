import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import json

from torchvision.transforms import Compose
import pandas as pd


class SiamFCDataset(Dataset):

    def __init__(self, dataset_path, num_pairs=10, transform=None):
        self.dataset_path = dataset_path
        self.video_dir_list = os.listdir(self.dataset_path)
        self.transform = transform
        self.num_pairs = num_pairs

    def __len__(self):
        return len(self.video_dir_list)

    def _filter(self, exist, gt):
        is_exist = np.where((exist[1:] == 1) & (exist[:-1] == 1))[0]  #choice exist[i] && exist[i+1]==1
        indices = [i for i, elem in enumerate(gt) if elem != [] and 200 <= elem[0] <= 300 and 250 <= elem[1] <= 350]
        indices = np.array(indices)
        common = np.intersect1d(is_exist, indices)
        common_next = common + 1
        valid_i = common[np.isin(common_next, common)]
        selected = np.random.choice(valid_i, self.num_pairs, replace=False)
        return selected

    def _load_search(self, imgs_dir, gt, selected_idx):
        imgs_search = []
        gt_search = []
        for idx in selected_idx:
            idx = idx + 1
            img_search = cv2.imread(os.path.join(imgs_dir, "{:04d}.jpg".format(idx)))
            bbox = gt[idx]
            x_center, y_center, w, h = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3]
            crop_x, crop_y = x_center - 127, y_center - 127
            crop_img = img_search[crop_y:crop_y + 255, crop_x:crop_x + 255]
            bbox_new_x = bbox[0] - crop_x  # 目标左上角 x
            bbox_new_y = bbox[1] - crop_y  # 目标左上角 y
            bbox_new_w = bbox[2]  # 宽度不变
            bbox_new_h = bbox[3]  # 高度不变
            bbox_new = [bbox_new_x, bbox_new_y, bbox_new_w, bbox_new_h]

            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            # cv2.imshow("crop_img", crop_img)
            # cv2.waitKey(0)
            crop_img = np.transpose(crop_img, (2, 0, 1))
            imgs_search.append(crop_img)
            gt_search.append(bbox_new)
        return np.array(imgs_search).astype(np.float16) / 255.0, np.array(gt_search).astype(np.float16)

    def _load_template(self, imgs_dir, gt, selected_idx):
        imgs_template = []
        gt_template = []
        template_size = 127
        half_size = template_size // 2
        for idx in selected_idx:
            img_template = cv2.imread(os.path.join(imgs_dir, "{:04d}.jpg".format(idx)))
            bbox = gt[idx]
            x_center, y_center, w, h = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3]
            crop_x, crop_y = x_center - half_size, y_center - half_size
            crop_img = img_template[crop_y:crop_y + template_size, crop_x:crop_x + template_size]
            bbox_new_x = half_size - w // 2
            bbox_new_y = half_size - h // 2
            bbox_new_w = w
            bbox_new_h = h
            bbox_new = [bbox_new_x, bbox_new_y, bbox_new_w, bbox_new_h]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            # cv2.imshow("crop_img", crop_img)
            # cv2.waitKey(0)
            crop_img = np.transpose(crop_img, (2, 0, 1))
            imgs_template.append(crop_img)
            gt_template.append(bbox_new)
        return np.array(imgs_template).astype(np.float16) / 255.0, np.array(gt_template).astype(np.float16)

    def _ToTensor(self, imgs, bbox):
        imgs = torch.from_numpy(imgs).float()
        bbox = torch.from_numpy(bbox).float()
        return imgs, bbox

    def __getitem__(self, idx):
        video_dir = self.video_dir_list[idx]
        imgs_dir = os.path.join(self.dataset_path, video_dir, r"infrared")
        imgs_list = os.listdir(imgs_dir)
        imgs_list.sort()
        anno_path = os.path.join(self.dataset_path, video_dir, r"infrared.json")
        with open(anno_path, "r", encoding="utf-8") as file:
            anno = json.load(file)
        exist = np.array(anno["exist"])
        gt = anno["gt_rect"]
        selected_idx = self._filter(exist, gt)
        imgs_search, box_search = self._load_search(imgs_dir, gt, selected_idx)
        imgs_template, box_template = self._load_template(imgs_dir, gt, selected_idx)
        return self._ToTensor(imgs_search, box_search), self._ToTensor(imgs_template, box_template)


if __name__ == "__main__":
    dataset = SiamFCDataset("data", 10)
    a = dataset[0]
    print(a)
