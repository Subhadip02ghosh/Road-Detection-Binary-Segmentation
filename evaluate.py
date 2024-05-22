import os
import time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import time
from data import CustomRoadData
# from loss import DiceBCELoss
from roadseg_nn import RoadSegNN
# from segnet import SegNet
from utils import seeding, epoch_time
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from skimage.metrics import structural_similarity as ssim


def evaluate_model(model, loader, device):
    model.eval()
    time_taken, mse, ssim, psnr, accuracy, recall, precision, f1 = 0., 0., 0., 0., 0., 0., 0., 0.
    total_img = 0.

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            total_img += x.shape[0]

            start_time = time.time()
            y_pred = model(x)   # Prediction
            y_pred = (y_pred >= 0.5)*1.
            end_time = time.time()
            time_taken += end_time - start_time

            # Ground Truth
            y_true = F.interpolate(
                y, (y_pred.size(-2), y_pred.size(-1)))

            # Mean Squared Error (MSE)
            mse += torch.mean((y_true - y_pred) ** 2)

            # Structural Similarity Index Measure (SSIM)
            c1 = 1e-3
            c2 = 3e-3
            mu1 = torch.mean(y_true, dim=-1)
            mu2 = torch.mean(y_pred, dim=-1)

            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = torch.mean(y_true**2, dim=-1) - mu1_sq
            sigma2_sq = torch.mean(y_true**2, dim=-1) - mu2_sq
            sigma12 = torch.mean(y_true*y_pred, dim=-1) - mu1_mu2

            ssim_n = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
            ssim_d = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
            ssim += ssim_n/ssim_d

            # Peak Signal-to-Noise Ratio (PSNR)
            if mse.item() == 0.:
                psnr += float('inf')
            else:
                PIXEL_MAX = 1.0
                psnr += 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

            # Accuracy
            accuracy += torch.sum((y_pred == y_true)*1.)

            # Recall (Sensitivity): Recall = True Positive (TP) / True Positive (TP) + False Negative (FN)
            TP = torch.sum(y_true & y_pred).item()
            FN = torch.sum(y_true & ~y_pred).item()

            recall += TP/(TP + FN)

            # Precision: Precision = True Positive (TP)/ [True Positive (TP) + False Positives(FP)]
            FP = torch.sum(~y_true & y_pred).item()
            precision += TP/(TP + FP)

            # F1 Score: F1 Score = 2 * ((Precision * Recall)/(Precision + Recall))
            f1 += 2 * ((precision * recall)/(precision + recall))

        time_taken /= total_img
        mse /= total_img
        ssim /= total_img
        psnr /= total_img
        accuracy /= total_img
        recall /= total_img
        precision /= total_img
        f1 /= total_img

    return [time_taken, mse, ssim, psnr, accuracy, recall, precision, f1]


if __name__ == "__main__":
    seeding(42)

    H = 256
    W = 256
    size = (H, W)
    batch_size = 16

    root_path = ".\\E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Satellite Data\\3) FINAL (CORRECTLY LABELLED)\\valid\\"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transforms_img = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(0.2826, 0.2029),
        ]
    )
    transforms_mask = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor()
        ]
    )

    """ Dataset and loader """
    valid_dataset = CustomRoadData(
        root_path, 'validate', transform_img=transforms_img, transform_mask=transforms_mask)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # ResNet-50
    model_type = 'ResNet-50'
    # os.makedirs(f'.\\Results\\{model_type}', exist_ok=True)
    model_path = f'E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Results\\{model_type}'
    model = RoadSegNN(backbone_type=model_type)
    model = model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    metrics = evaluate_model(model, valid_loader, device)
    # Printing: MSE, SSIM, PSNR, Accuracy, Recall, Precision, F1 Score
    print("RoadSegNN (ResNet-50): ",
          f'MSE: {metrics[0]}, SSIM: {metrics[1]}, PSNR: {metrics[2]}, Accuracy: {metrics[3]}, Recall: {metrics[4]}, Precision: {metrics[5]}, F1 Score: {metrics[6]}, ')
    del model
    torch.cuda.empty_cache()

    # ResNet-101
    model_type = 'ResNet-101'
    # os.makedirs(f'.\\Results\\{model_type}', exist_ok=True)
    model_path = f'E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Results\\{model_type}'
    model = RoadSegNN(backbone_type=model_type)
    model = model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    metrics = evaluate_model(model, valid_loader, device)
    # Printing: MSE, SSIM, PSNR, Accuracy, Recall, Precision, F1 Score
    print("RoadSegNN (ResNet-101): ",
          f'MSE: {metrics[0]}, SSIM: {metrics[1]}, PSNR: {metrics[2]}, Accuracy: {metrics[3]}, Recall: {metrics[4]}, Precision: {metrics[5]}, F1 Score: {metrics[6]}, ')
    del model
    torch.cuda.empty_cache()

    # Swin-T
    model_type = 'Swin-T'
    # os.makedirs(f'.\\Results\\{model_type}', exist_ok=True)
    model_path = f'E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Results\\{model_type}'
    model = RoadSegNN(backbone_type=model_type)
    model = model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    metrics = evaluate_model(model, valid_loader, device)
    # Printing: MSE, SSIM, PSNR, Accuracy, Recall, Precision, F1 Score
    print("RoadSegNN (Swin-T): ",
          f'MSE: {metrics[0]}, SSIM: {metrics[1]}, PSNR: {metrics[2]}, Accuracy: {metrics[3]}, Recall: {metrics[4]}, Precision: {metrics[5]}, F1 Score: {metrics[6]}, ')
    del model
    torch.cuda.empty_cache()

    # # SegNet
    # model_type = 'SegNet'
    # # os.makedirs('.\\Results\\Segnet', exist_ok=True)
    # model_path = f'E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Results\\{model_type}'
    # model = SegNet()
    # model = model.load_weights(os.path.join(model_path, "ckpt.pth"))
    # model = model.to(device)
    # metrics = evaluate_model(model, valid_loader, model_path, device)
    # # Printing: MSE, SSIM, PSNR, Accuracy, Recall, Precision, F1 Score
    # print("SegNet: ",
    #       f'MSE: {metrics[0]}, SSIM: {metrics[1]}, PSNR: {metrics[2]}, Accuracy: {metrics[3]}, Recall: {metrics[4]}, Precision: {metrics[5]}, F1 Score: {metrics[6]}, ')
    # del model
    # torch.cuda.empty_cache()
