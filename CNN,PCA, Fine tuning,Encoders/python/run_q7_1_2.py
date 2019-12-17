# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:45:10 2019

@author: Barathwaj
"""

#https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/ reference

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


device = torch.device("cpu")
max_iters = 9
batch_size = 32
learning_rate = 0.1
hidden_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

trainset= torchvision.datasets.MNIST(root = './data' , train =  True,download = True, transform= transform)
testset = torchvision.datasets.MNIST(root = './data' , train =  False,download = True, transform= transform)


train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True)

test_loader = DataLoader(testset, batch_size=batch_size,shuffle=False)




class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(14*14*2, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*2)
        x = self.fc1(x)
        return x
            
    
model = ConvNet()


trainLoss_list = []
trainAcc_list = []

optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate )

for itrs in range(max_iters):
    total_loss = 0
    total_acc = 0
    correct_prediction =0
    for trainXY in train_loader:
        x = torch.tensor(trainXY[0])

        
        y = torch.tensor(trainXY[1])
 
        #targets = torch.max(y, 1)[1]
        
        model
        probs = model(x)
        
        
        loss = torch.nn.functional.cross_entropy(probs,y)
        
        
        total_loss += loss.item()
        probss = torch.max(probs, 1)[1]
        correct_prediction = correct_prediction  + probss.eq(y.data).cpu().sum().item()
         
         
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    acc = correct_prediction/len(trainset)
    trainLoss_list.append(total_loss)
    trainAcc_list.append(acc)
    
    if itrs % 2 == 0:
        print("itrs: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itrs, total_loss, acc))
        
        






correct_predictions_test = 0
for data in test_loader:
    # get the inputs
    inputs = torch.autograd.Variable(data[0])
    labels = torch.autograd.Variable(data[1])
    

    # get output
    model
    y_pred = model(inputs)
    loss = nn.functional.cross_entropy(y_pred, labels)

    predicted = torch.max(y_pred, 1)[1]
    correct_predictions_test += torch.sum(predicted == labels.data).item()

test_acc = correct_predictions_test/len(trainset)

print('Test accuracy: {}'.format(test_acc))
        

plt.figure(1)
plt.plot(np.arange(1,10), trainAcc_list )
#plt.plot(np.arange(1,101), validation_accList )
plt.gca().legend(('train_accuracy'))
plt.title("epoch vs accuracy ")
plt.xlabel("epoch")
plt.ylabel("accuracy")

#plt.figure(2)
#plt.plot(np.arange(1,101), validation_accList )


plt.figure(3)
plt.plot(np.arange(1,10), trainLoss_list )
plt.title("trainData_loss")
plt.xlabel("epoch")
plt.ylabel("loss")

         
         
         
                
        