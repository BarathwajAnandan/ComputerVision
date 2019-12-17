# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:34:51 2019

@author: KevinX
"""
from run_q4 import *
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


import os
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation



learning_rate = 0.01
hiddensize = 64
batch_size = 64
max_iters = 4
d =0
if d ==1:
    device = 'cuda'
else :
    device = 'cpu'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
train_x = torchvision.datasets.EMNIST(root = '../data', split = 'balanced', transform = transform,train = True, download = True)
test_x = torchvision.datasets.EMNIST(root = '../data', split = 'balanced', transform = transform,train = False, download = True)


train_loader = torch.utils.data.DataLoader(train_x, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_x, batch_size=batch_size, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(stride=2, kernel_size=2))
        self.fc1 = nn.Sequential(nn.Linear(7*7*32, 1024),nn.ReLU(),nn.Linear(1024, 47))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 7*7*32)
        x = self.fc1(x)
        return x    
        

model = ConvNet()


trainLoss_list = []
trainAcc_list = []
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



for itrs in range(max_iters):
    total_loss = 0
    total_acc = 0
    correct_prediction =0
    for trainXY in train_loader:
        x = torch.tensor(trainXY[0]).to(device)
        
        x.shape

        
        y = torch.tensor(trainXY[1]).to(device)
 
        #targets = torch.max(y, 1)[1]
        
        model.to(device)
        
        probs = model(x)
        
        
        loss = torch.nn.functional.cross_entropy(probs,y)
        
        
        total_loss += loss.item()
        probss = torch.max(probs, 1)[1]
        correct_prediction = correct_prediction  + probss.eq(y.data).to(device).sum().item()
        
        acc = correct_prediction/len(train_x)
         
         
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
   
    trainLoss_list.append(total_loss)
    trainAcc_list.append(acc)
    
    if itrs % 2 == 0:
        print("itrs: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itrs, total_loss, acc))
        
        
        
    
    
    
plt.figure(1)
plt.plot(np.arange(1,5), trainAcc_list )
#plt.plot(np.arange(1,101), validation_accList )
plt.title("epoch vs accuracy ")
plt.xlabel("epoch")
plt.ylabel("accuracy")

#plt.figure(2)
#plt.plot(np.arange(1,101), validation_accList )






plt.figure(3)
plt.plot(np.arange(1,5), trainLoss_list )
plt.title("trainData_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
        
        
        
        
        
        
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        #rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                #fill=False, edgecolor='red', linewidth=2)
        #plt.gca().add_patch(rect)
    #plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    num_letters = len(bboxes)
    height = bw.shape[0]
    width = bw.shape[1]

    letters = []
    row_index = np.zeros((num_letters, 1))
    h_ = []
    w_ = []

    line_no = 0
    bboxes_sorted = []
    y2_last = 0
    for i,bbox in enumerate(bboxes):
        
        y1, x1, y2, x2 = bbox
        letter = bw[y1:y2+1, x1:x2+1]
        
        height = letter.shape[0]
        width = letter.shape[1]
        h_.append(height)
        w_.append(width)
        
        #print(y2-y1)
        if y2 - y2_last >= y2 - y1:
            line_no += 1
        y2_last = y2
        row_index[i] = line_no
        bboxes_sorted.append((y1, x1, y2
                              , x2, line_no))
    h_ = np.asarray(h_)
    w_ = np.asarray(w_)
    avg_w = np.sum(w_)/num_letters
    avg_h = np.sum(h_)/num_letters
    bboxes_sorted = sorted(bboxes_sorted, key=lambda x: (x[4], x[-2])) #key lambda used for sorting elements based on 2 and 5th element
    for i,bbox in enumerate(bboxes_sorted):
        
        y1, x1, y2, x2, row_id = bbox
        letter = bw[y1:y2, x1:x2]
        

        #print(y2,x2,y1,x1)
        #plt.imshow(letter)
        #plt.show()
        
        #print(letter.shape)
        

# =============================================================================
#         y1 -=  bbox_padding_y
#         x1 -=  bbox_padding_x
#         
#         y2 +=   bbox_padding_y
#         x2 +=  bbox_padding_x
#         
# =============================================================================
        
        
        
        #x1 = np.pad(x2,20, mode = 'symmetric' )

        #letter = bw[y1:y2+1, x1:x2+1]
        letter = np.pad(letter,(20,20),'maximum')
        letter = skimage.morphology.binary_erosion(letter)
        letter = skimage.filters.gaussian(letter,sigma = 1)






#Transforming image as per the dataset images for right predictions! , size is 28,28 and inverse .
        letter = skimage.transform.resize(letter,(28,28)).T      
        letter = 1-letter
        letters.append(letter)
    input_data = np.asarray(letters)        
        
        
    
    
    
    #test_x = np.array([np.reshape(x, (32, 32)) for x in letters])

    tss = []
    for item in input_data:
        ts = transform(np.expand_dims(item, axis=2)).type(torch.float32)
        tss.append(ts)
    x_ts = torch.stack(tss, dim=0).to(device)
    
    

        
    # get output
    model.to(device)
    probs = model(x_ts)

    predicted = torch.max(probs, 1)[1]

    

    # ground_truth = np.argmax(test_y, axis=1)
    predicted = torch.max(probs, 1)[1].cpu().numpy()
    # print(predicted)
    import string
    letter_list = np.array([str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]] + [_ for _ in string.ascii_lowercase[:11]])
    predicted_letters =  letter_list[predicted]

    num_extracted_letters = predicted_letters.shape[0]
    
    curr_row = ""
    line_no = 0
    for i in range(num_extracted_letters):

        if row_index[i] == line_no:
            curr_row += predicted_letters[i]
        else:
            print(curr_row)
            curr_row = ""+ predicted_letters[i]
            line_no = row_index[i]
    print(curr_row)



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
