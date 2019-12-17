import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words
from matplotlib import pyplot as plt
import cv2

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    labels  = train_data['labels']
    K = dictionary.shape[0]
    layer_num = 2
    histograms = []
 
    pathh = train_data['files']
    
    
    labels  = train_data['labels']
    for count,path in enumerate(pathh) :        
#    if (count ==2):
#        break               
#        print(count, path)
# his_path = plt.imread("../data/" + path)
        file_path = "../data/" + path
        histogram =  get_image_feature(file_path, dictionary, layer_num,K)  
        
# =============================================================================
#         
#         arg_2 = [(file_path, dictionary, layer_num,K) for count,path in enumerate(pathh)]
#         (file_path, dictionary, layer_num,K) = arg_2
#         with multiprocessing.Pool(os.cpu_count()) as p:
#             histogram =  p.map(get_image_feature,(file_path, dictionary, layer_num,K))
#             histograms = np.append(histograms,histogram)
#         
#         
#         
# 
# =============================================================================

     
        histograms = np.append(histograms,histogram)
        #histograms = np.vstack((histograms,histo))
    #histograms = np.reshape(histograms,(pathh.shape[0],dict_size))
 #       print (histograms.shape)
        print (count)
    histograms = histograms.reshape((len(train_data['labels']), histogram.shape[0]))
    #print (histograms.shape)
    
    
    np.savez("trained_system",features = histograms,labels = labels,dictionary =dictionary,SPM_layer_num = layer_num)
    

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    
    
    test_labels = test_data['labels']
    test_images = test_data['files']
    trained_system = np.load("trained_system.npz")
    
    trained_system_features = trained_system['features']
    trained_system_labels = trained_system['labels']
    trained_system_dictionary =  trained_system['dictionary']
    trained_system_SPM_layer_num  = trained_system['SPM_layer_num']
    
    l = []
    count = 0
    Confusion_Matrix = np.zeros((8,8)) 
    for i,path in enumerate(test_images):
        
        test_path = "../data/" + path
        
        img = plt.imread(test_path)
        img = img.astype('float')/255.0
        
        word_map = visual_words.get_visual_words(img,trained_system_dictionary)
        
        test_hist = get_feature_from_wordmap_SPM(word_map,trained_system_SPM_layer_num,trained_system_dictionary.shape[0])
        
        similarity = distance_to_set(test_hist,trained_system_features)
        
        predicted_label = np.argmax(similarity)
        
        p =trained_system_labels[predicted_label]
        print ( trained_system_labels[predicted_label], "    " , test_labels[i])
        l.append(p)
        
        if p == test_labels[i]:
            count = count +1
        print(count, "/",i+1)
        Confusion_Matrix[test_labels[i],p] += 1
        

    accuracy = np.diag(Confusion_Matrix).sum()/Confusion_Matrix.sum()

    return Confusion_Matrix ,accuracy        


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

        

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    image = skimage.io.imread(file_path)

    image = image.astype('float')/255

    wordmap = visual_words.get_visual_words(image,dictionary)
    
    histo =  get_feature_from_wordmap_SPM(wordmap,layer_num,K)

    return histo 

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''  
    
    similarity = np.sum(np.minimum(word_hist,histograms),axis = 1)
   # print (similarity.shape)
    return similarity

def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    
    '''        
    hist,bins = np.histogram(wordmap,bins = dict_size,density = True)    
    return hist,bins

def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    
    W = []
    hist_all = []
    
    for i in range(0,layer_num+1):
        if i == 0 or i ==1 :
           # print(layer_num)
           
            #print (layer_num)
            te = 1/(2**layer_num)
           # print (te)
            W.append(te)
           # print (i,W[i])
        else:
            te = 1/2**(-i+layer_num+1)
            W.append(te)
        if i == 0:
            hist,bins = get_feature_from_wordmap(wordmap,dict_size)
            hist_all = np.append(hist_all,hist*W[i])
            
         #   print("Hello",hist_all.shape)
        elif i ==1 :
            
           
            split_4 = np.array_split(wordmap,2**i,axis = 0)
            for j in split_4:
                
                first_cut =   np.array_split(wordmap,2**i,axis = 1)
                for final_cut in first_cut:
                
                    hist,bins = get_feature_from_wordmap(final_cut,dict_size)
                    hist_all = np.append(hist_all,hist*W[i])
             
        else:
           # print (wordmap.shape)
           # print ("third Start")
            split_16 = np.array_split(wordmap,2**(i),axis = 0)
            for k in split_16 :
                first_cut = np.array_split(wordmap,2**i,axis =1)
                for final_cut in first_cut:
                    hist, bins = get_feature_from_wordmap(final_cut,dict_size)
                    hist_all = np.append(hist_all,hist*W[i])

    return hist_all/sum(hist_all)
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
