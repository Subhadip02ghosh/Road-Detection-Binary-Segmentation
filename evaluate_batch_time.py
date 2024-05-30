import os
import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import time
from data import CustomRoadData
from roadseg_nn import RoadSegNN
from segnet import SegNet
from utils import seeding, epoch_time


def evaluate_model_batch_time(model, loader, device):
    model.eval()
    time_taken = 0.
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

        time_taken /= total_img
        
    return time_taken


if __name__ == "__main__":
    seeding(42)

    H = 256
    W = 256
    size = (H, W)
    batch_size = 16

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
    batch_time = evaluate_model_batch_time(model, valid_loader, device)
    with open(f'{model_path}\\Evaluation_Batch_Time.txt', 'w') as f:
        f.write(f'Time taken \(Batch\): {batch_time}\n')

    # Printing: Time taken (Batch)
    print("RoadSegNN (ResNet-50):", f'Time taken \(Batch\): {batch_time}')
    del model
    torch.cuda.empty_cache()

    # ResNet-101
    model_type = 'ResNet-101'
    model_path = f'.\\Results\\{model_type}'
    model = RoadSegNN(backbone_type=model_type)
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    batch_time = evaluate_model_batch_time(model, valid_loader, device)
    with open(f'{model_path}\\Evaluation_Batch_Time.txt', 'w') as f:
        f.write(f'Time taken \(Batch\): {batch_time}\n')

    # Printing: Time taken (Batch)
    print("RoadSegNN (ResNet-101):", f'Time taken \(Batch\): {batch_time}')
    del model
    torch.cuda.empty_cache()

    # Swin-T
    model_type = 'Swin-T'
    model_path = f'.\\Results\\{model_type}'
    model = RoadSegNN(backbone_type=model_type)
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    batch_time = evaluate_model_batch_time(model, valid_loader, device)
    with open(f'{model_path}\\Evaluation_Batch_Time.txt', 'w') as f:
        f.write(f'Time taken \(Batch\): {batch_time}\n')

    # Printing: Time taken (Batch)
    print("RoadSegNN (Swin-T):", f'Time taken \(Batch\): {batch_time}')
    del model
    torch.cuda.empty_cache()

    # SegNet
    model_type = 'SegNet'
    model_path = f'.\\Results\\{model_type}'
    model = SegNet()
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    batch_time = evaluate_model_batch_time(model, valid_loader, device)
    with open(f'{model_path}\\Evaluation_Batch_Time.txt', 'w') as f:
        f.write(f'Time taken \(Batch\): {batch_time}\n')
        
    # Printing: Time taken (Batch)
    print("SegNet:", f'Time taken \(Batch\): {batch_time}')
    del model
    torch.cuda.empty_cache()
