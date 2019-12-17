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


device = torch.device("cpu")

train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x = train_data['train_data']
test_x = test_data['test_data']

train_y = train_data['train_labels']
test_y = test_data['test_labels']

max_iters = 100
batch_size = 64
learning_rate = 0.4
hidden_size = 64
train_xTensor = torch.from_numpy(train_x).float().to(device)
train_yTensor = torch.from_numpy(train_y).to(device)

test_xTensor = torch.from_numpy(test_x).float().to(device)
test_yTensor = torch.from_numpy(test_y).to(device)

train_loader = DataLoader(TensorDataset(train_xTensor, train_yTensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_xTensor, test_yTensor), batch_size=batch_size, shuffle=False)

D_in = train_x.shape[1]
H = hidden_size
D_out = train_y.shape[1]
class TwoLayernetwork(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayernetwork,self).__init__()
        self.fc1 = nn.Linear(D_in,H)
        self.fc2 = nn.Linear(H,D_out)
    def forward(self,X):
        
        X = torch.sigmoid(self.fc1(X))
        #X = torch.softmax(X,dim = -1)
        X = self.fc2(X)
        return X              
model = TwoLayernetwork(D_in,H,D_out)

model.to(device)

trainLoss_list = []
trainAcc_list = []

optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate )

for itrs in range(max_iters):
    total_loss = 0
    total_acc = 0
    correct_prediction =0
    for trainXY in train_loader:
        x = torch.autograd.variable(trainXY[0])
        y = torch.autograd.variable(trainXY[1])
        targets = torch.max(y, 1)[1]
        
        
        probs = model(x)
        
        
        loss = torch.nn.functional.cross_entropy(probs,targets)
        
        
        total_loss += loss.item()
        probss = torch.max(probs, 1)[1]
        correct_prediction = correct_prediction  + probss.eq(targets.data).cpu().sum().item()
         
         
        loss.backward()
        optimizer.step()
        
        optimizer.zero_grad()
    acc = correct_prediction/train_y.shape[0]
    trainLoss_list.append(total_loss)
    trainAcc_list.append(acc)
    
    if itrs % 2 == 0:
        print("itrs: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itrs, total_loss, acc))
        

correct_predictions_test = 0
for x,y in test_loader:
    # get the inputs
    x = torch.tensor(x).float()
    y = torch.tensor(y)
    targets = torch.max(y, 1)[1]

    # get output
    y_pred = model(x)
    loss = nn.functional.cross_entropy(y_pred, targets)

    predicted = torch.max(y_pred, 1)[1]
    correct_predictions_test += torch.sum(predicted == targets.data).item()

test_acc = correct_predictions_test/test_y.shape[0]

print('Test accuracy: {}'.format(test_acc))
        


#for plotting the Accuracy and loss graphs
#import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(np.arange(1,101), trainAcc_list )
#plt.plot(np.arange(1,101), validation_accList )
plt.gca().legend(('train_accuracy'))
plt.title("epoch vs accuracy ")
plt.xlabel("epoch")
plt.ylabel("accuracy")

#plt.figure(2)
#plt.plot(np.arange(1,101), validation_accList )


plt.figure(3)
plt.plot(np.arange(1,101), trainLoss_list )
plt.title("trainData_loss")
plt.xlabel("epoch")
plt.ylabel("loss")

# =============================================================================
# plt.plot(np.arange(1,101), loss )
# plt.title("epoch vs Loss (learningrate*10 (0.03))")
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# =============================================================================
             
        