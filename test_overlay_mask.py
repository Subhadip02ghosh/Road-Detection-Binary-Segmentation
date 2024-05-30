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
import matplotlib.pyplot as plt


def img_to_tensor(img_path):
    transform = transforms.Compose([transforms.PILToTensor()])
    with Image.open(img_path) as f:
        f.convert('L')
        img_tensor = transform(f)
        print(img_tensor)
        # ANIKET DA: I was getting an error that the model expects float, but PILToTensor returns uint8 by default.
        # So I type cast the tensor to float.
        img_tensor = img_tensor.float()
        return img_tensor


if __name__ == "__main__":
    img_path = "E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Satellite Data\\3) FINAL (CORRECTLY LABELLED)\\Final - Transformed Data (Augmented)\\train\\sat\\36.jpg"
    # img_path = "E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Satellite Data\\3) FINAL (CORRECTLY LABELLED)\\Final - Transformed Data (Augmented)\\valid\\sat\\6125.jpg"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    H = 256
    W = 256
    size = (H, W)

    # For input to the Model (The image is being normalised)
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(0.2826, 0.2029)
        ]
    )

    # For viewing/displaying the sat image with mask overlay
    transform2 = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor()
        ]
    )

    img = transform(Image.open(img_path).convert('L')).unsqueeze(0)
    x = img.to(device, dtype=torch.float32)

    """ MODEL INIT """
    # model_type = 'ResNet-50'
    # model_type = 'ResNet-101'
    # model_type = 'Swin-T'
    model_type = 'SegNet'

    # model_path = f"E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Results_Old\\Results\\{model_type}"
    # model_path = f"E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Results (Retrained - 30 May)\\Results\\{model_type}"
    model_path = f"E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Final Results\\Results\\{model_type}"

    # model = RoadSegNN(backbone_type=model_type)
    model = SegNet()

    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)
        y = (y >= 0.5)*1.
        # print(y)

    # PRINTING IMAGE
    # Re-Reading image for viewing

    # img = transform2(Image.open(img_path).convert('L')).unsqueeze(0)
    # x2 = img.to(device, dtype=torch.float32)
    # print(x2[0].shape)
    # x2 = x2[0].permute(1, 2, 0).cpu().numpy()
    # y = y[0].permute(1, 2, 0).cpu().numpy()

    # Sat image reopened for viewing
    x2 = np.array(Image.open(img_path).convert('L'))
    y = F.interpolate(y, (x2.shape[0], x2.shape[1]))
    y = y[0].permute(1, 2, 0).cpu().numpy()

    plt.figure()
    plt.imshow(x2, cmap='gray')
    plt.imshow(y, alpha=0.5)
    plt.axis('off')
    plt.show()
