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


def evaluate_model(model, loader, device):
    model.eval()
    time_taken, mse, ssim, psnr, accuracy, TP, FN, FP = 0., 0., 0., 0., 0., 0, 0, 0
    total_img = 0.

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            total_img += x.shape[0]

            start_time = time.time()
            y_pred = model(x)   # Prediction
            y_pred = y_pred >= 0.5
            end_time = time.time()
            time_taken_loop = end_time - start_time
            time_taken += time_taken_loop

            # Ground Truth
            y_true = F.interpolate(
                y, (y_pred.size(-2), y_pred.size(-1))).bool()

            # Mean Squared Error (MSE)
            mse_loop = torch.mean((y_true*1. - y_pred*1.) ** 2)
            mse += mse_loop

            # Structural Similarity Index Measure (SSIM)
            c1 = 1e-3
            c2 = 3e-3
            mu1 = torch.mean(y_true*1., dim=[1, 2, 3])
            mu2 = torch.mean(y_pred*1., dim=[1, 2, 3])

            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = torch.mean((y_true*1.)**2, dim=[1, 2, 3]) - mu1_sq
            sigma2_sq = torch.mean((y_true*1.)**2, dim=[1, 2, 3]) - mu2_sq
            sigma12 = torch.mean((y_true*1.)*(y_pred*1.),
                                 dim=[1, 2, 3]) - mu1_mu2

            ssim_n = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
            ssim_d = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
            ssim_loop = torch.mean(ssim_n/ssim_d)
            ssim += ssim_loop
            # Peak Signal-to-Noise Ratio (PSNR)
            if mse_loop.item() == 0.:
                psnr_loop = float('inf')
                psnr += psnr_loop
            else:
                PIXEL_MAX = 1.0
                psnr_loop = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse_loop))
                psnr += psnr_loop

            # Accuracy
            accuracy_loop = torch.sum(y_true == y_pred)/torch.numel(y_true)
            accuracy += accuracy_loop

            TP_loop = torch.sum(y_true & y_pred).item()
            TP += TP_loop
            FN_loop = torch.sum(y_true & ~y_pred).item()
            FN += FN_loop
            FP_loop = torch.sum(~y_true & y_pred).item()
            FP += FP_loop
            print(f'\rIteration [{i+1:02}/{len(loader)}] ==> Time_taken: {time_taken_loop:.4f} s | Accuracy: {accuracy_loop:.4f} | TP: {TP_loop} | FN: {FN_loop} | FP: {FP_loop}', end='\r')
        print()

        time_taken /= total_img
        accuracy /= len(loader)
        mse /= len(loader)
        ssim /= len(loader)
        psnr /= len(loader)

        # Recall (Sensitivity): Recall = True Positive (TP) / True Positive (TP) + False Negative (FN)
        recall = TP/(TP + FN + 1e-7)

        # Precision: Precision = True Positive (TP)/ [True Positive (TP) + False Positives(FP)]
        precision = TP/(TP + FP + 1e-7)

        # F1 Score: F1 Score = 2 * ((Precision * Recall)/(Precision + Recall))
        f1 = 2 * ((precision * recall)/(precision + recall + 1e-7))

    return [time_taken, mse, ssim, psnr, accuracy, recall, precision, f1]


if __name__ == "__main__":
    seeding(42)

    H = 256
    W = 256
    size = (H, W)
    batch_size = 1

    root_path = ".\\transformed_data"

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
    model_path = f'.\\Results\\{model_type}'
    model = RoadSegNN(backbone_type=model_type)
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    metrics = evaluate_model(model, valid_loader, device)
    with open(f'{model_path}\\Evaluation.txt', 'w') as f:
        f.write(
            f'Time taken: {metrics[0]}\nMSE: {metrics[1]}\nSSIM: {metrics[2]}\nPSNR: {metrics[3]}\nAccuracy: {metrics[4]}\nRecall: {metrics[5]}\nPrecision: {metrics[6]}\nF1 Score: {metrics[7]}\n')
    # Printing: Time taken, MSE, SSIM, PSNR, Accuracy, Recall, Precision, F1 Score
    print("RoadSegNN (ResNet-50): ",
          f'Time taken: {metrics[0]}, MSE: {metrics[1]}, SSIM: {metrics[2]}, PSNR: {metrics[3]}, Accuracy: {metrics[4]}, Recall: {metrics[5]}, Precision: {metrics[6]}, F1 Score: {metrics[7]}')
    del model
    torch.cuda.empty_cache()

    # ResNet-101
    model_type = 'ResNet-101'
    model_path = f'.\\Results\\{model_type}'
    model = RoadSegNN(backbone_type=model_type)
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    metrics = evaluate_model(model, valid_loader, device)
    with open(f'{model_path}\\Evaluation.txt', 'w') as f:
        f.write(
            f'Time taken: {metrics[0]}\nMSE: {metrics[1]}\nSSIM: {metrics[2]}\nPSNR: {metrics[3]}\nAccuracy: {metrics[4]}\nRecall: {metrics[5]}\nPrecision: {metrics[6]}\nF1 Score: {metrics[7]}\n')
    # Printing: Time taken, MSE, SSIM, PSNR, Accuracy, Recall, Precision, F1 Score
    print("RoadSegNN (ResNet-101): ",
          f'Time taken: {metrics[0]}, MSE: {metrics[1]}, SSIM: {metrics[2]}, PSNR: {metrics[3]}, Accuracy: {metrics[4]}, Recall: {metrics[5]}, Precision: {metrics[6]}, F1 Score: {metrics[7]}')
    del model
    torch.cuda.empty_cache()

    # Swin-T
    model_type = 'Swin-T'
    model_path = f'.\\Results\\{model_type}'
    model = RoadSegNN(backbone_type=model_type)
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    metrics = evaluate_model(model, valid_loader, device)
    with open(f'{model_path}\\Evaluation.txt', 'w') as f:
        f.write(
            f'Time taken: {metrics[0]}\nMSE: {metrics[1]}\nSSIM: {metrics[2]}\nPSNR: {metrics[3]}\nAccuracy: {metrics[4]}\nRecall: {metrics[5]}\nPrecision: {metrics[6]}\nF1 Score: {metrics[7]}\n')
    # Printing: Time taken, MSE, SSIM, PSNR, Accuracy, Recall, Precision, F1 Score
    print("RoadSegNN (Swin-T): ",
          f'Time taken: {metrics[0]}, MSE: {metrics[1]}, SSIM: {metrics[2]}, PSNR: {metrics[3]}, Accuracy: {metrics[4]}, Recall: {metrics[5]}, Precision: {metrics[6]}, F1 Score: {metrics[7]}')
    del model
    torch.cuda.empty_cache()

    # SegNet
    model_type = 'SegNet'
    model_path = f'.\\Results\\{model_type}'
    model = SegNet()
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    metrics = evaluate_model(model, valid_loader, device)
    with open(f'{model_path}\\Evaluation.txt', 'w') as f:
        f.write(
            f'Time taken: {metrics[0]}\nMSE: {metrics[1]}\nSSIM: {metrics[2]}\nPSNR: {metrics[3]}\nAccuracy: {metrics[4]}\nRecall: {metrics[5]}\nPrecision: {metrics[6]}\nF1 Score: {metrics[7]}\n')
    # Printing: Time taken, MSE, SSIM, PSNR, Accuracy, Recall, Precision, F1 Score
    print("SegNet: ",
          f'Time taken: {metrics[0]}, MSE: {metrics[1]}, SSIM: {metrics[2]}, PSNR: {metrics[3]}, Accuracy: {metrics[4]}, Recall: {metrics[5]}, Precision: {metrics[6]}, F1 Score: {metrics[7]}')
    del model
    torch.cuda.empty_cache()
