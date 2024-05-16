import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np

training_dataset_path = "C:\\Users\\tiudrdo1\\Desktop\\ROadSeg_4thyr\\transformed_data\\sat\\"

def img_loader(image):
    f = Image.open(image).convert('L')
    return f

class CustomDataset(Dataset):
    def __init__(self, training_dataset_path, transform=None):
        
        self.transform = transform
        self.sat_path = os.path.join(training_dataset_path)
        self.sat_imgs = os.listdir(self.sat_path)
        self.n_samples = len(self.sat_imgs)

    def __getitem__(self, index):
        image = img_loader(os.path.join(self.sat_path,self.sat_imgs[index]))
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return self.n_samples


# NOTE: The size set here needs to be same as the size set in the Dataloader Transforms,
# Since while the Mean remains the same, the Std. Deviation changes depending on dimensions
training_transforms = transforms.Compose(
    [transforms.Resize((512, 512)), transforms.ToTensor()]
)

train_dataset = CustomDataset(training_dataset_path, transform=training_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_dataset.__len__(), shuffle=False)


def get_mean_and_std(loader):
    for images in loader:
        mean = images.mean()
        std = images.std()

    return mean, std


mean, std = get_mean_and_std(train_loader)
print(mean, std)