# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:45:10 2019

@author: Barathwaj
"""
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt



device = 'cpu'





train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x = train_data['train_data']
test_x = test_data['test_data']

train_y = train_data['train_labels']
test_y = test_data['test_labels']



max_iters = 50
batch_size = 64
learning_rate = 0.01
hidden_size = 64


train_x = np.array([np.reshape(x, (32, 32)) for x in train_x])
test_x = np.array([np.reshape(x, (32, 32)) for x in test_x])

train_xTensor = torch.from_numpy(train_x).float().unsqueeze(1).to(device)
train_yTensor = torch.from_numpy(train_y).to(device)

test_xTensor = torch.from_numpy(test_x).float().unsqueeze(1).to(device)
test_yTensor = torch.from_numpy(test_y).to(device)

train_loader = DataLoader(TensorDataset(train_xTensor, train_yTensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_xTensor, test_yTensor), batch_size=batch_size, shuffle=False)




class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(16*16*8, 1024),nn.ReLU(),nn.Linear(1024, 36))

    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape)
        x = x.view(-1, 16*16*8)
        x = self.fc1(x)
        return x
            
print(device)  
model = ConvNet()


trainLoss_list = []
trainAcc_list = []

optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate )

for itr in range(max_iters):
    total_loss = 0
    correct = 0
    for data in train_loader:
        # get the inputs
        inputs = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])
        targets = torch.max(labels, 1)[1]
        
        # get output
        
        model.to(device)
        y_pred = model(inputs)
        loss = nn.functional.cross_entropy(y_pred, targets)

        total_loss += loss.item()
        predicted = torch.max(y_pred, 1)[1]
        correct += torch.sum(predicted == targets.data).item()

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    acc = correct/train_x.shape[0]
    trainLoss_list.append(total_loss)
    trainAcc_list.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, acc))
        
        







correct_predictions_test = 0
for data in test_loader:
    # get the inputs
    inputs = torch.autograd.Variable(data[0])
    labels = torch.autograd.Variable(data[1])
    targets = torch.max(labels, 1)[1]

    # get output
    
    model.to(device)
    y_pred = model(inputs)
    loss = nn.functional.cross_entropy(y_pred, targets)

    predicted = torch.max(y_pred, 1)[1]
    correct_predictions_test += torch.sum(predicted == targets.data).item()

test_acc = correct_predictions_test/test_y.shape[0]

print('Test accuracy: {}'.format(test_acc))
        



#for plotting the Accuracy and loss graphs
#import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(np.arange(1,51), trainAcc_list )
#plt.plot(np.arange(1,101), validation_accList )
plt.gca().legend(('train_accuracy'))
plt.title("epoch vs accuracy ")
plt.xlabel("epoch")
plt.ylabel("accuracy")

#plt.figure(2)
#plt.plot(np.arange(1,101), validation_accList )






plt.figure(3)
plt.plot(np.arange(1,51), trainLoss_list )
plt.title("trainData_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
         
         
                
        