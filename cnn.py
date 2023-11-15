import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn
import torch.nn.functional as F
import tqdm
from options import *

def train_cnn_model(num_samples, train_split,num_epochs,learning_rate, batch_size):
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # print(num_epochs)
    # Load MNIST dataset
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Split the dataset
    train_size = int(num_samples * train_split)
    indices = list(range(num_samples))
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)

    # Define your neural network model, loss function, optimizer, and training loop here
    # (model definition and training code would go here)
    class CNNMnist(nn.Module):
        def __init__(self):
            super(CNNMnist, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    model = CNNMnist()
    # print(mlp)  # net architecture

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # optimize all logistic parameters
    criterion = torch.nn.NLLLoss()
    epoch_loss = []
    epoch_accuracy = []
    for i in range(num_epochs):
        batch_loss = []
        total, correct = 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
    

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            # if batch_idx % 50 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1, batch_idx * len(images), len(trainloader.dataset),
            #         100. * batch_idx / len(trainloader), loss.item()))
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        loss_avg = sum(batch_loss)/len(batch_loss)
        # print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)
        epoch_accuracy.append(correct/total)
    return epoch_loss, epoch_accuracy
    
    torch.save(model.state_dict(), 'model_cnn.pth')


