import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import time
from data import CustomRoadData
from loss import DiceBCELoss, CombinedLoss
from roadseg_nn import RoadSegNN
from segnet import SegNet
from utils import seeding, epoch_time

def trainer(model, loader, optimizer, loss_fn, device, epoch):
    epoch_loss = 0.0
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        y = F.interpolate(y, (y_pred.size(-2), y_pred.size(-1)))
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        data_str = f'\rEpoch: {epoch+1:02} | Iteration: {i+1:02} | Loss: {loss.item():.4f}'
        print(data_str, end='\r')
    print()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            y = F.interpolate(y, (y_pred.size(-2), y_pred.size(-1)))
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

# Main Training Function, To Be Called
def train(model, train_loader, valid_loader, optimizer, checkpoint_path, device):
    model.train()
    best_valid_loss = float("inf")
    loss_fn = CombinedLoss()
    # loss_fn = DiceBCELoss()

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = trainer(model, train_loader, optimizer, loss_fn, device, epoch)
        with open(os.path.join(checkpoint_path, 'train_loss.txt'), 'a') as f:
            f.write(f'Epoch[{epoch+1}/{num_epochs}]:\t{str(train_loss)}\n')
        if epoch % 10 == 0.:
            valid_loss = evaluate(model, valid_loader, loss_fn, device)
            with open(os.path.join(checkpoint_path, 'valid_loss.txt'), 'a') as f:
                f.write(f'Epoch[{epoch+1}/{num_epochs}]:\t{str(valid_loss)}\n')

            """ Saving the model """
            if valid_loss <= best_valid_loss:
                data_str = f"Valid loss improved/remained same from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {os.path.join(checkpoint_path, 'ckpt.pth')}"
                print(data_str)

                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(checkpoint_path, 'ckpt.pth'))

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str = data_str + f'\t Val. Loss: {valid_loss:.3f}\n' if epoch % 10 == 0. else data_str
        print(data_str)

if __name__ == "__main__":
    seeding(1)

    os.makedirs('.\\Results', exist_ok=True)
    root_path = ".\\transformed_data"

    """ Hyperparameters """
    H = 256
    W = 256
    size = (H, W)
    batch_size = 16
    num_epochs = 300
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
    train_dataset = CustomRoadData(root_path, 'train', transform_img=transforms_img, transform_mask=transforms_mask)
    valid_dataset = CustomRoadData(root_path, 'validate', transform_img=transforms_img, transform_mask=transforms_mask)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,        
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # ResNet-50
    backbone_type='ResNet-50'
    os.makedirs(f'.\\Results\\{backbone_type}', exist_ok=True)
    checkpoint_path = f'.\\Results\\{backbone_type}'
    model = RoadSegNN(backbone_type=backbone_type)
    model = model.to(device)
    lr = 1e-3
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train(model, train_loader, valid_loader, optimizer, checkpoint_path, device)
    del model, optimizer
    torch.cuda.empty_cache()

    # ResNet-101
    backbone_type='ResNet-101'
    os.makedirs(f'.\\Results\\{backbone_type}', exist_ok=True)
    checkpoint_path = f'.\\Results\\{backbone_type}'
    model = RoadSegNN(backbone_type=backbone_type)
    model = model.to(device)
    lr = 1e-3
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train(model, train_loader, valid_loader, optimizer, checkpoint_path, device)
    del model, optimizer
    torch.cuda.empty_cache()

    # Swin-T
    backbone_type='Swin-T'
    os.makedirs(f'.\\Results\\{backbone_type}', exist_ok=True)
    checkpoint_path = f'.\\Results\\{backbone_type}'
    model = RoadSegNN(backbone_type=backbone_type)
    model = model.to(device)
    lr = 5e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    train(model, train_loader, valid_loader, optimizer, checkpoint_path, device)
    del model, optimizer
    torch.cuda.empty_cache()

    # # SegNet
    # os.makedirs('.\\Results\\Segnet', exist_ok=True)
    # checkpoint_path = '.\\Results\\Segnet'
    # model = SegNet()
    # model = model.to(device)
    # lr = 1e-3
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # train(model, train_loader, valid_loader, optimizer, checkpoint_path, device)
    # del model, optimizer
    # torch.cuda.empty_cache()
    