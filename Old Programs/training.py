import torchvision
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import segnet

# import os
# import matplotlib.pyplot as plt
# import numpy as np

train_dataset_path = ""  # TODO # add path
test_dataset_path = ""  # TODO # add path


mean = [0.4444, 0.4441, 0.4454]  # TODO  # if necessary  # placeholders
std = [0.2222, 0.2122, 0.2252]  # TODO # if necessary  # placeholders

train_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        # Additional augmentations/transformations
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ]
)


# Dataset Access at File Directory
train_dataset = torchvision.datasets.ImageFolder(
    root=train_dataset_path, transform=train_transforms
)
test_dataset = torchvision.datasets.ImageFolder(
    root=test_dataset_path, transform=test_transforms
)

# DataLoader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False)


# To display transformed images

# def show_transformed_images(dataset):
#     loader = torch.utils.data.DataLoader(dataset, batch_size = 6, shuffle = True)
#     batch = next(iter(loader))
#     images, labels = batch

#     grid = torchvision.util.make_grid(images, nrow = 3)
#     plt.figure(figsize = (11, 11))
#     plt.imshow(np.transpose(grid, (1, 2, 0)))
#     print('labels: ', labels)

# show_transformed_images(train_dataset)


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)


# Evaluation Function
def evaluate_classifier_on_test_set(classifier, test_loader):
    classifier.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    # Prevents Back Propagation. Speeds up the Computations for the Evaluation of the NN on the Test Set
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)  # Correct Labels from Test Set

            # Keeps track of batch size, useful for last batch where no. of images != stated Batch Size always
            total += labels.size(0)

            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)

            predicted_correctly_on_epoch += (predicted == labels).sum().item()

    # Test Dataset accuracy for this epoch
    epoch_accuracy = predicted_correctly_on_epoch / total * 100.00
    # Formatting to 3 decimal places
    epoch_accuracy = "{:.3f}".format(epoch_accuracy)

    # Printing Evaluation Diagnostics
    print(
        f"    - Testing dataset. Correctly Classified Images: {predicted_correctly_on_epoch} out of {total}. Epoch Accuracy: {epoch_accuracy}."
    )

    return epoch_accuracy


def save_checkpoint(classifier, epoch, optimizer, best_accuracy):
    state = {
        'epoch': epoch + 1,
        'classifier': classifier.state_dict(),
        'best_accuracy': best_accuracy,
        'optimizer': optimizer.state_dict(),
        'comments': f"Best Checkpoint: Epoch {epoch}, with Highest Accuracy: {best_accuracy}%."
    }
    torch.save(state, 'classifier_best_checkpoint.pth.tar')


# Function: Train Neural Network
def train_nn(classifier, train_loader, test_loader, loss_func, optimizer, n_epochs):
    device = set_device()
    best_accuracy = 0  # Saves the best accuracy

    for epoch in range(n_epochs):
        print(f"Epoch no.: {epoch + 1}")
        classifier.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = classifier(images)  # Classifying images here

            _, predicted = torch.max(outputs.data, 1)

            # Calculate loss. outputs = model's classified road, labels = true road
            loss = loss_func(outputs, labels)

            loss.backward()  # Backpropagation for weight updation
            optimizer.step()  # Update weight
            running_loss += loss.items()  # Update Running Loss

            # No. of Correctly Predicted Images in the current iteration
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_correct / total * 100.00
        # Formatting to 3 decimal places
        epoch_loss = "{:.3f}".format(epoch_loss)
        epoch_accuracy = "{:.3f}".format(epoch_accuracy)

        # Printing Training Diagnostics
        print(
            f"    - Training dataset. Correctly Classified Images: {running_correct} out of {total}. Epoch Accuracy: {epoch_accuracy}. Epoch Loss: {epoch_loss}."
        )

        test_data_accuracy = evaluate_classifier_on_test_set(
            classifier, test_loader)
        if (test_data_accuracy > best_accuracy):
            best_accuracy = test_data_accuracy
            save_checkpoint(classifier, epoch, optimizer, best_accuracy)

        print("Training Finished!")
        return classifier


# Training
classifier = segnet.SegNet(1)
num_features = 0  # TODO # placeholder # if necessary
num_classes = 2
device = set_device()
classifier = classifier.to(device)
loss_func = nn.CrossEntropyLoss()

optimizer = optim.SGD(classifier, lr=0.01, momentum=0.9, weight_decay=0.003)
train_nn(classifier, train_loader, test_loader,
         loss_func, optimizer, epoch=150)


# Print Checkpoint Details
checkpoint = torch.load('classifier_best_checkpoint.pth.tar')
print(checkpoint['epoch'])
print(checkpoint['best accuracy'])
print(checkpoint['comments'])

# Load Checkpoint
# TODO: IF NECESSARY, Need to implement a checkpoint loading code for the models.
