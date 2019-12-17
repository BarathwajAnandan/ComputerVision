import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']
test_x = test_data['test_data']
dim = 32
# do PCA
##########################
##### your code here #####
##########################
x_mean = np.mean(train_x,axis = 0)
x = train_x - x_mean
data = x.T@x
U,S,V = np.linalg.svd(data)

projection_matrix = U[:,:dim]

recon = x @ projection_matrix @ projection_matrix.T




valid_mean = np.mean(valid_x,axis = 0)

valid_ = valid_x - valid_mean

recon_valid = valid_ @ projection_matrix @ projection_matrix.T


recon_valid +=valid_mean


# visualize the comparison and compute PSNR
##########################
##### your code here #####
##########################



x_ = []


fig, axs = plt.subplots(10, 2, sharex=True, sharey=True)
for count,i in enumerate([0,1,101,110,3501,3530,2510,2530,1601,1665]):
    x = valid_x[i].reshape(32,32).T
    axs[count,0].imshow(x)
    x_.append(x)
    
test_mean = np.mean(test_x,axis = 0)

test_ = test_x - test_mean

recon_test = test_ @ projection_matrix @ projection_matrix.T


recon_test +=test_mean
  

    
    
for count,i in enumerate([0,1,101,110,3501,3530,2510,2530,1601,1665]):
    x = recon_valid[i].reshape(32,32).T
    axs[count,1].imshow(x)
    x_.append(x)



total_psnr = 0
for j in range(0,(valid_x.shape[0])):
    p = psnr(valid_x[i],recon_valid[i])
    
    total_psnr += p
average_psnr = total_psnr/valid_x.shape[0]