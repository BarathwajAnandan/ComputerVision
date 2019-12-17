import numpy as np
import scipy.ndimage
import os
import util
import torch
import deep_recog
import skimage
import matplotlib.pyplot as plt
def extract_deep_feature(x, vgg16_weights):
    
    
    weights = util.get_VGG16_weights()
    
  #preprocessing:
  
    x = x.astype('float') / 255
    x = skimage.transform.resize(x, (224, 224, 3), preserve_range= True )


    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    x = (x - mean) / std
    
    count = 0
 
    for l in weights:   
        
    
        
# =============================================================================
#         if c ==10:
#             break
# =============================================================================
        if l[0] == 'conv2d' :
            print ("Convolution Time")
            x = multichannel_conv2d(x,l[1],l[2])
        if count ==2:
            print("breakinnn")  
            break
            

        if l[0] == 'relu':
            print ("Relu Time!")
            x = relu(x)
            
      
            
        if l[0] == 'maxpool2d':
            print ("Pool Time!")
            x = max_pool2d(x,l[1])  
            
           
    
        if l[0] == 'linear':
            print ("Linear Time! entering once")
            
            count+= 1
            
            if count ==1:
                x = x.T
                x = np.einsum('kli->kil', x)
                
            
            x = linear(x,l[1],l[2])

         
    print("image is : " , x)
    print(x.shape)
#    np.save("convo",x)
    return x
    
    #print(x.shape)

    
   # print ("shape : " , x.shape)
    
    
    
    

    

def multichannel_conv2d(x, weight, bias):
    h,w,c = x.shape
    in_channels_K  = x.shape[2]
  #  print (in_channels_K)
    
    out_channels_J = weight.shape[0]
 #   print (out_channels_J)
    
    con_out = np.zeros((h,w,out_channels_J))
    #cor_out = np.zeros((h,w,out_channels_J))
    
    for J in range(0,out_channels_J):
        for K in range(0,in_channels_K):
            
            
            
            flip_weight = np.flip (weight[J,K,:,:] , axis = 1)
            flip_weight = np.flip (flip_weight , axis = 0)
            #print("J and K are ", J , K , " :  ",flip_weight.shape)
            #print(flip_weight)
            #filters = scipy.ndimage.correlate(x[:,:,K],weight[J,K,:,:] , mode = 'constant')
            filters = scipy.ndimage.convolve(x[:,:,K],flip_weight , mode = 'constant')  
            #cor_out[:,:,J] += filters
            con_out[:,:,J] += filters
            
  #  print ("conv",con_out +bias)
  #  print ("corr", cor_out+ bias)
        
   #print("conv: " , con_out.shape)
    #print(flip_weight.shape)  
    
    return con_out + bias
         
    
    
def relu(x):
    
   
    x = np.maximum(x, 0)    
    print ("Relu :",x.shape)
    
    return x


# reference - Stackoverflow   
def max_pool2d(x, size):
    m, n = x.shape[:2]
    ky = kx= size
    ny=m//ky
    nx=n//kx
    x_pad=x[:ny*ky, :nx*kx, ...]
    new_shape=(ny,ky,nx,kx)+x.shape[2:]
    x=np.nanmax(x_pad.reshape(new_shape),axis=(1,3))     
    return x




def linear(x,W,b):
    
    print(x.shape)
    
    
   
    print(x.shape)
    
   # if (b):
    #    b = False

    x = x.flatten()    
    x = np.matmul(W,x) + b
    print ("Linear :       " ,x.shape)
    return x 
    
    '''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)

	'''
#TEST
path_img = "C:/Users/KevinX/Desktop/Computer Vision/HW1/data/aquarium/sun_apetcnxozfplpysg.jpg"
image =  plt.imread(path_img)
x = extract_deep_feature(image, 5)

