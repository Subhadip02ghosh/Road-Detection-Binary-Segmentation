# This code augments the dataset, by making 9 additional versions of each sat-mask pair.

import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from PIL import Image
import cv2

def img_loader(image):
    with Image.open(image) as f:
        f.convert('L')
        return np.array(f)
    
def mask_loader(mask):
    return cv2.imread(mask, 0)

class CustomDataset(Dataset):
    def __init__(self, root_path):

        self.sat_path = os.path.join(root_path, "sat")
        self.sat_imgs = os.listdir(self.sat_path)
        self.mask_path = os.path.join(root_path, "mask")
        self.mask_imgs = os.listdir(self.mask_path)
        self.n_samples = len(self.sat_imgs)

    def __getitem__(self, index):
        mask = mask_loader(os.path.join(self.mask_path,self.mask_imgs[index]))
        image = img_loader(os.path.join(self.sat_path,self.sat_imgs[index]))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        mask = mask_loader(os.path.join(self.mask_path,self.mask_imgs[index]))
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        return image, mask

    def __len__(self):
        return self.n_samples

def geometric_transform(images, masks, degrees=30., translate=(0.5, 0.5), scale=(0.5, 1.5), flip_h=0.75, flip_v=0.75):
    angles = random.uniform(-degrees, degrees)
    translate_x = random.uniform(-translate[0], translate[0]) * images.size(-1)
    translate_y = random.uniform(-translate[1], translate[1]) * images.size(-2)
    scale_factors = random.uniform(scale[0], scale[1])
    hori = random.uniform(0, 1)<=flip_h
    vert = random.uniform(0, 1)<=flip_v
    
    transformed_images = F.affine(images, angles, translate=(translate_x, translate_y), scale=scale_factors, shear =0.)
    transformed_masks = F.affine(masks, angles, translate=(translate_x, translate_y), scale=scale_factors, shear =0.)
    
    if hori:
        transformed_images = F.hflip(transformed_images)
        transformed_masks = F.hflip(transformed_masks)
    if vert:
        transformed_images = F.vflip(transformed_images)
        transformed_masks = F.vflip(transformed_masks)

    return transformed_images, transformed_masks

def save_image(image, directory, name):
    os.makedirs(directory, exist_ok=True)
    plt.imsave(os.path.join(directory, name), image, cmap='gray', vmin=0, vmax=255)

with torch.no_grad():
    dataset = CustomDataset(root_path='Data')
    num_augmentations = 9
    batch_size = dataset.__len__()
    dataloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    count = 0
    for i, (batch_images, batch_masks) in enumerate(dataloader):
        for k in range(batch_size):
            imageNumpy = batch_images[k].squeeze(0).long().numpy()
            maskNumpy = batch_masks[k].squeeze(0).long().numpy()
            save_image(imageNumpy, 'transformed_data/image', f'{count}.jpg')
            save_image(maskNumpy, 'transformed_data/mask', f'{count}.png')
            count += 1
        for j in range(num_augmentations):
            transformed_images, transformed_masks = geometric_transform(batch_images.cuda(), batch_masks.cuda())
            for k in range(batch_size):
                image = transformed_images[k].squeeze(0).cpu().long().numpy()
                mask =transformed_masks[k].squeeze(0).cpu().long().numpy()

                save_image(image, 'transformed_data/image', f'{count}.jpg')
                save_image(mask, 'transformed_data/mask', f'{count}.png')
                count += 1