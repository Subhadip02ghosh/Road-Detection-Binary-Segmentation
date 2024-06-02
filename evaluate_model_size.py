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


def evaluate_model_parameters(model, device):
    model.eval()

    num_parameters = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)

    return num_parameters


if __name__ == "__main__":
    seeding(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ResNet-50
    model_type = 'ResNet-50'
    # model_path = f'.\\Results\\{model_type}'
    model_path = f"E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Final Results\\Results\\{model_type}"
    model = RoadSegNN(backbone_type=model_type)
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    num_parameters = evaluate_model_parameters(model, device)
    print(f"RoadSegNN (ResNet-50) has {num_parameters} parameters")
    del model
    torch.cuda.empty_cache()

    # ResNet-101
    model_type = 'ResNet-101'
    # model_path = f'.\\Results\\{model_type}'
    model_path = f"E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Final Results\\Results\\{model_type}"
    model = RoadSegNN(backbone_type=model_type)
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    num_parameters = evaluate_model_parameters(model, device)
    print(f"RoadSegNN (ResNet-101) has {num_parameters} parameters")
    del model
    torch.cuda.empty_cache()

    # Swin-T
    model_type = 'Swin-T'
    # model_path = f'.\\Results\\{model_type}'
    model_path = f"E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Final Results\\Results\\{model_type}"
    model = RoadSegNN(backbone_type=model_type)
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    num_parameters = evaluate_model_parameters(model, device)
    print(f"RoadSegNN (Swin-T) has {num_parameters} parameters")
    del model
    torch.cuda.empty_cache()

    # SegNet
    model_type = 'SegNet'
    # model_path = f'.\\Results\\{model_type}'
    model_path = f"E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Model and Loss\\Final Results\\Results\\{model_type}"
    model = SegNet()
    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    num_parameters = evaluate_model_parameters(model, device)
    print(f"SegNet() has {num_parameters} parameters")
    del model
    torch.cuda.empty_cache()
