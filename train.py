import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from data import CustomRoadData
from loss import DiceBCELoss
from osrd import OSRD
from segnet import SegNet
from utils import seeding, epoch_time

def trainer(model, loader, optimizer, loss_fn):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(model.device, dtype=torch.float32)
        y = y.to(model.device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(model.device, dtype=torch.float32)
            y = y.to(model.device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

# Main Training Function, To Be Called
def train(model, train_loader, valid_loader, checkpoint_path):
    model.train()
    best_valid_loss = float("inf")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = DiceBCELoss()

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = trainer(model, train_loader, optimizer, loss_fn)
        if epoch % 10 == 0.:
            valid_loss = evaluate(model, valid_loader, loss_fn)

            """ Saving the model """
            if valid_loss < best_valid_loss:
                data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
                print(data_str)

                best_valid_loss = valid_loss
                torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)



if __name__ == "__main__":
    seeding(42)

    os.makedirs('.\\Results', exist_ok=True)
    root_path = ".\\Data"

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 16
    num_epochs = 300
    lr = 1e-3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ]
)

    """ Dataset and loader """
    train_dataset = CustomRoadData(root_path, 'train',transform=train_transforms)
    valid_dataset = CustomRoadData(root_path, 'validate')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # ResNet-50
    backbone_type='ResNet-50'
    os.makedirs(f'.\\Results\\{backbone_type}', exist_ok=True)
    checkpoint_path = os.path.join(f'.\\Results\\{backbone_type}', 'ckpt.pth')
    model = OSRD(backbone_type=backbone_type)
    model = model.to(device)
    train(model, train_loader, valid_loader, checkpoint_path)
    del model
    torch.cuda.empty_cache()

    # ResNet-101
    backbone_type='ResNet-101'
    os.makedirs(f'.\\Results\\{backbone_type}', exist_ok=True)
    checkpoint_path = os.path.join(f'.\\Results\\{backbone_type}', 'ckpt.pth')
    model = OSRD(backbone_type=backbone_type)
    model = model.to(device)
    train(model, train_loader, valid_loader, checkpoint_path)
    del model
    torch.cuda.empty_cache()

    # Swin-T
    backbone_type='Swin-T'
    os.makedirs(f'.\\Results\\{backbone_type}', exist_ok=True)
    checkpoint_path = os.path.join(f'.\\Results\\{backbone_type}', 'ckpt.pth')
    model = OSRD(backbone_type=backbone_type)
    model = model.to(device)
    train(model, train_loader, valid_loader, checkpoint_path)
    del model
    torch.cuda.empty_cache()

    # SegNet
    os.makedirs('.\\Results\\Segnet', exist_ok=True)
    checkpoint_path = os.path.join('.\\Results\\Segnet', 'ckpt.pth')
    model = SegNet()
    model = model.to(device)
    train(model, train_loader, valid_loader, checkpoint_path)
    del model
    torch.cuda.empty_cache()
    


