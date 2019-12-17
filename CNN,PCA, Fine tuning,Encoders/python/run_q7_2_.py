# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:54:07 2019

@author: KevinX
"""

#from run_q4 import *
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

device = torch.device('cpu')

train_dir = '../data/oxford-flowers17/train'
val_dir = '../data/oxford-flowers17/val'
test_dir = '../data/oxford-flowers17/test'

batch_size = 32
max_iters = 70
learning_rate = 0.001

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([ transforms.CenterCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
trainset = ImageFolder(train_dir, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True)



val_test_transform = transforms.Compose([ transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
valset = ImageFolder(val_dir, transform=val_test_transform)
val_loader = DataLoader(valset,batch_size=batch_size,shuffle=False)


testset = ImageFolder(test_dir, transform=val_test_transform)
test_loader = DataLoader(testset, batch_size=batch_size,shuffle=False)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 6, kernel_size=7, stride=1, padding=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(28*28*120, 84),
                                 nn.ReLU()
                                 )
        self.fc2 = nn.Sequential(nn.Linear(84, 17))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 28*28*120)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def calc_accuracy(loader):
    # Set the model to eval mode
    model.eval()
    correct_predictions, total_predictions = 0, 0
    for x, y in train_loader:
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).to(device)

        # forward
        
        model.cuda()
        y_pred = model(x)
        predicted = torch.max(y_pred, 1)[1]
        correct_predictions += torch.sum(predicted == y.data).item()
        total_predictions += x.size(0)

    # Calculate the accuracy
    acc = correct_predictions/total_predictions
    return acc


model = NN()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loss = []
train_acc = []
val_acc = []
for itr in range(0, max_iters):
    total_loss = 0
    correct = 0
    model.train()
    for x, y in train_loader:
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).to(device)

        # get output
        model.to(device)
        y_pred = model(x)
        loss = nn.functional.cross_entropy(y_pred, y)

        total_loss += loss.item()
        predicted = torch.max(y_pred, 1)[1]
        correct += torch.sum(predicted == y.data).item()

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    total_predictions = len(trainset) 
    acc = correct/total_predictions
    train_loss.append(total_loss)
    train_acc.append(acc)

    valid_acc = calc_accuracy(val_loader)
    val_acc.append(valid_acc)

    print("itr: {:02d} \t train_loss: {:.4f} \t train_acc: {:.4f} \t val_acc: {:.4f}".format(itr, total_loss, acc, valid_acc))

    if itr == 100:
        learning_rate /= 10
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# checkpoint = torch.load('q7_2_scratch_model_parameter200.pkl')
# model.load_state_dict(checkpoint)

test_acc = calc_accuracy(test_loader)
print("Test Accuracy: {:.4f}".format(test_acc))



plt.figure(1)
plt.plot(np.arange(1,71), train_acc[1:71] )
#plt.plot(np.arange(1,101), validation_accList )
plt.gca().legend(('train_accuracy'))
plt.title("epoch vs accuracy ")
plt.xlabel("epoch")
plt.ylabel("accuracy")

#plt.figure(2)
#plt.plot(np.arange(1,101), validation_accList )


plt.figure(3)
plt.plot(np.arange(1,71), train_loss [1:71])
plt.title("trainData_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
