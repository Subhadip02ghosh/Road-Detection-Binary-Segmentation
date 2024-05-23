import os
import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import CustomRoadData
from roadseg_nn import RoadSegNN
from segnet import SegNet
from utils import seeding, epoch_time
from PIL import Image
import cv2
import torchvision.transforms as transforms
import numpy as np
from roadseg_nn import RoadSegNN
from segnet import SegNet
import matplotlib as plt


def img_to_tensor(image):
    transform = transforms.Compose([transforms.PILToTensor()])
    with Image.open(image) as f:
        f.convert('L')
        img_tensor = transform(f)
        return img_tensor


if __name__ == "__main__":

    img_path = "E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Satellite Data\\3) FINAL (CORRECTLY LABELLED)\\valid\\sat\\6125.jpg"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ResNet-50
    model_type = 'ResNet-50'
    model_path = f"E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Results\\{model_type}"
    model = RoadSegNN(backbone_type=model_type)
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    model.eval()

    x = img_to_tensor(img_path)

    with torch.no_grad():
        y = model(x)
        y = (y >= 0.5) * 1.
        y = y[0].transpose(1, 2, 0).cpu().numpy()
        x = x[0].transpose(1, 2, 0).cpu().numpy()

        plt.imshow(x, cmap='gray')
        plt.imshow(y, cma='gray', alpha=0.6)
        plt.show()
