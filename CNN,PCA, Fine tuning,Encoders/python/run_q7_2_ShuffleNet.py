# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:56:12 2019

@author: KevinX
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt

device = torch.device('cuda')

train_dir = '../data/oxford-flowers17/train'
val_dir = '../data/oxford-flowers17/val'
test_dir = '../data/oxford-flowers17/test'

batch_size = 32
num_workers = 4
num_epochs1 = 20

learning_rate1 = 0.001
learning_rate2 = 0.00001

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([ transforms.CenterCrop(256),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
trainset = ImageFolder(train_dir, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True)



val_test_transform = transforms.Compose([ transforms.CenterCrop(256),transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
valset = ImageFolder(val_dir, transform=val_test_transform)
val_loader = DataLoader(valset,batch_size=batch_size,shuffle=False)


testset = ImageFolder(test_dir, transform=val_test_transform)
test_loader = DataLoader(testset, batch_size=batch_size,shuffle=False)




# getting the model
model = torchvision.models.squeezenet1_1(pretrained=True)
classCount = len(trainset.classes)


#modifying the classfier to give right number of classes
model.classifier[1] = nn.Conv2d(512, classCount, kernel_size=(1,1))
model.num_classes = classCount

# =============================================================================
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.classifier.parameters():
#     param.requires_grad = True
# =============================================================================






optimizer = torch.optim.SGD(model.classifier.parameters(), lr=learning_rate1)
print("A")
train_accs = []
train_losses = []
for epoch in range(num_epochs1):
    # Run an epoch over the training data.
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).to(device)

        # forward
        
        model.to(device)
        y_pred = model(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        total_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check accuracy on the train and val sets.
    model.eval()
    num_correct, num_samples = 0, 0
    for x, y in train_loader:
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).to(device)

        # forward
        model.to(device)
        y_pred = model(x)
        predicted = torch.max(y_pred, 1)[1]
        num_correct += torch.sum(predicted == y.data).item()
        num_samples += x.size(0)

    # Calculate the accuracy
    train_acc = num_correct/num_samples
    
    
    
    
    model.eval()
    num_correct, num_samples = 0, 0
    for x, y in val_loader:
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).to(device)

        # forward
        model.to(device)
        y_pred = model(x)
        predicted = torch.max(y_pred, 1)[1]
        num_correct += torch.sum(predicted == y.data).item()
        num_samples += x.size(0)

    # Calculate the accuracy
    valid_acc = num_correct/num_samples

    train_accs.append(train_acc)
    train_losses.append(total_loss)

    print("itr: {:02d} \t train_loss: {:.4f} \t train_acc: {:.4f} \t valid_acc: {:.4f}".format(epoch, total_loss, train_acc, valid_acc))
    
    
    if epoch == 10 :
        print(": New Lrate!!")
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=learning_rate1)
        lr = learning_rate2
        
for param in model.parameters():
    param.requires_grad = True




plt.figure(1)
plt.plot(np.arange(1,21), train_accs )
#plt.plot(np.arange(1,101), validation_accList )

plt.title("epoch vs accuracy ")
plt.xlabel("epoch")
plt.ylabel("accuracy")

#plt.figure(2)
#plt.plot(np.arange(1,101), validation_accList )


plt.figure(3)
plt.plot(np.arange(1,21), train_losses )
plt.title("trainData_loss")
plt.xlabel("epoch")
plt.ylabel("loss")



# checkpoint = torch.load('q7_2_finetune_model_parameter.pkl')
# model.load_state_dict(checkpoint)

model.eval()
num_correct, num_samples = 0, 0
for x, y in test_loader:
    x = torch.tensor(x).float().to(device)
    y = torch.tensor(y).to(device)
    model.to(device)
    # forward
    y_pred = model(x)
    predicted = torch.max(y_pred, 1)[1]
    num_correct += torch.sum(predicted == y.data).item()
    num_samples += x.size(0)

    # Calculate the accuracy
    test_acc = num_correct/num_samples
print("Test Accuracy: {:.4f}".format(test_acc))