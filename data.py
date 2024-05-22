import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


def img_loader(image):
    f = Image.open(image).convert('L')
    return f


def mask_loader(image):
    f = Image.open(image).convert('1')
    return f


class CustomRoadData(Dataset):
    def __init__(self, root_path, mode='train', transform_img=None, transform_mask=None):

        self.transform_img = transform_img
        self.transform_mask = transform_mask
        if mode == 'validate':
            self.sat_path = os.path.join(root_path, "valid\\sat")
            self.sat_imgs = os.listdir(self.sat_path)
            self.mask_path = os.path.join(root_path, "valid\\mask")
            self.mask_imgs = os.listdir(self.mask_path)
            self.n_samples = len(self.sat_imgs)
        else:
            self.sat_path = os.path.join(root_path, "train\\sat")
            self.sat_imgs = os.listdir(self.sat_path)
            self.mask_path = os.path.join(root_path, "train\\mask")
            self.mask_imgs = os.listdir(self.mask_path)
            self.n_samples = len(self.sat_imgs)

    def __getitem__(self, index):
        """ Reading image """
        image = img_loader(os.path.join(self.sat_path, self.sat_imgs[index]))
        if self.transform_img:
            image = self.transform_img(image)

        """ Reading mask """
        mask = mask_loader(os.path.join(self.mask_path, self.mask_imgs[index]))
        if self.transform_mask:
            mask = self.transform_mask(mask)
        return image, mask

    def __len__(self):
        return self.n_samples
