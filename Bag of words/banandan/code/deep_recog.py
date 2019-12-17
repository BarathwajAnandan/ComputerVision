import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms as transforms
import util
#import network_layers
import matplotlib.pyplot as plt
import torch.nn as nn

def build_recognition_system(vgg16, num_workers=2):
    
     train_data = np.load("../data/train_data.npz")
     labels = train_data['labels']
     features = np.empty((0,4096))     

#   image_feature = get_image_feature((5,"aquarium/sun_apetcnxozfplpysg.jpg",vgg16))
#   print(image_feature)
#   print(image_feature.shape)       
#   image_feature = get_image_feature((5,image_path,vgg16))
#     print("shape of Feature:", image_feature.shape)       
     for i,path in enumerate(train_data['files']):
         print (i)  
         image_feature = get_image_feature((5,path,vgg16))
         features = np.append(features,image_feature.detach().numpy(),axis = 0)
         print(features[0].shape)        
     print (features.shape)  
     print("zip created")
     np.savez("trained_system_deep.npz", features = features , labels = labels )
'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N, K)
	* labels: numpy.ndarray of shape (N)
	'''
def evaluate_recognition_system(vgg16, num_workers=2):
    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system_deep.npz")    
    test_labels = test_data['labels']
    test_images = test_data['files']    
    trained_system_features = trained_system['features']
    trained_system_labels = trained_system['labels']   
        
    count = 0
    Confusion_Matrix = np.zeros((8,8)) 

	
    for i,path in enumerate(test_images):
        test_feature = get_image_feature((5,path,vgg16))  
        test_feature = test_feature.detach().cpu().numpy()                 
        predicted_label  = np.argmax(distance_to_set(test_feature,trained_system_features))              
        p =trained_system_labels[predicted_label]                
        if p == test_labels[i]:
            count = count +1
        print(count, "/",i+1)
        Confusion_Matrix[test_labels[i],p] += 1        
    return Confusion_Matrix
# =============================================================================
# 	'''
# 	Evaluates the recognition system for all test images and returns the confusion matrix.
# 
# 	[input]
# 	* vgg16: prebuilt VGG-16 network.
# 	* num_workers: number of workers to process in parallel
# 
# 	[output]
# 	* conf: numpy.ndarray of shape (8, 8)
# 	* accuracy: accuracy of the evaluated system
# 	'''
#        
# =============================================================================

def preprocess_image(image):

#stackoverflow    
    image = image.astype('float') / 255
    image = skimage.transform.resize(image, (224, 224, 3), preserve_range= True )


    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    image = (image - mean) / std


    
    return image
def get_image_feature(args):
    
    '''
    	Extracts deep features from the prebuilt VGG-16 network.
        s is a function run by a subprocess.
            put]
    	* i: index of training image
        mage_path: path of image file
        gg16: prebuilt VGG-16 network.
	
	[output]
	* feat: evaluated deep feature
    '''
    i, image_path, vgg16 = args
  
    image_torch = plt.imread("../data/" +image_path)
                                                                #preprocessing for pytorch
    image_torch = preprocess_image(image_torch)    
    image_torch = np.einsum('kli->ikl', image_torch)
    image_torch = image_torch[np.newaxis, :]
    image_torch =  torch.from_numpy(image_torch)
    image_torch = image_torch
    
    feature = vgg16.forward(image_torch)

    return feature

def distance_to_set(feature, train_features):
# =============================================================================
# 	'''
# 	Compute distance between a deep feature with all training image deep features.
# 
# 	[input]
# 	* feature: numpy.ndarray of shape (K)
# 	* train_features: numpy.ndarray of shape (N, K)
# 
# 	[output]
# 	* dist: numpy.ndarray of shape (N)
# 	'''
# =============================================================================
    neg_Euc =  np.linalg.norm(feature - train_features,axis = 1)   
    return -neg_Euc
	