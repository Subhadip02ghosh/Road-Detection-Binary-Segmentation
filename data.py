import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class CustomRoadData(Dataset):
    def __init__(self, root_path, mode='train', transform=None):

        if mode == 'validate':
            self.sat_path = os.path.join(root_path, "valid\\sat")
            self.sat_imgs = os.listdir(self.sat_path)
            self.mask_path = os.path.join(root_path, "valid\\mask")
            self.mask_imgs = os.listdir(self.mask_path)
            self.n_samples = len(self.sat_imgs)
            self.transform = None
        else:
            self.sat_path = os.path.join(root_path, "train\\sat")
            self.sat_imgs = os.listdir(self.sat_path)
            self.mask_path = os.path.join(root_path, "train\\mask")
            self.mask_imgs = os.listdir(self.mask_path)
            self.n_samples = len(self.sat_imgs)
            self.transform = transform

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(os.path.join(self.sat_path,self.sat_imgs[index]), 0)
        image = image/255.0 ## (512, 512, 3)
        image = np.expand_dims(image, axis=0) ## (1, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)

        """ Reading mask """
        mask = cv2.imread(os.path.join(self.mask_path,self.mask_imgs[index]), 0)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        if self.transform:
            mask = self.transform(mask)
        return image, mask

    def __len__(self):
        return self.n_samples