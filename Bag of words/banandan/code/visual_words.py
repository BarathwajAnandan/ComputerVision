import numpy as np
import multiprocessing

import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import matplotlib.pyplot as plt
import util
import random


def extract_filter_responses(img):
    
    #img = plt.imread(path_img)
    #img = img.astype('float')/255.0
 
    if img.shape[2] == 1 :
        img = np.tile(img,3)

    img = skimage.color.rgb2lab(img)
#l,a,b = cv2.split(img)
    l = img[:,:,0]
    a = img[:,:,1]
    b = img[:,:,2]

    filter_responses = []
    [filter_responses.append([]) for i in range(0,60)]

        
    scaled_values = [1,2,4,8,8*np.sqrt(2)]    
    print 
    for i,scale in enumerate(scaled_values):
                ''' Gaussian'''
                for j,ch in enumerate([l,a,b]):    
                    filter_responses[12*i +(0*3+j)]   = scipy.ndimage.gaussian_filter(ch,scale)
                    filter_responses[12*i +(1*3+j)]   = scipy.ndimage.filters.gaussian_laplace(ch,scale)
                    filter_responses[12*i +(2*3+j)]   = scipy.ndimage.filters.gaussian_filter1d(ch, scale,order = 1,axis=1)
                    filter_responses[12*i +(3*3+j)]   = scipy.ndimage.filters.gaussian_filter1d(ch, scale,order = 1,axis=0)

    filter_responses = np.asarray(filter_responses)
    #print (filter_responses.shape)
    new = np.einsum('kli->lik', filter_responses)
    #print (new.shape)
    
    return new
def get_visual_words(image, dictionary):
        
    
    filter_response = extract_filter_responses(image)
    h,w,c = filter_response.shape
    filter_response = filter_response.reshape(h*w,c)    
    g = scipy.spatial.distance.cdist(filter_response,dictionary,'euclidean')
    wordmap = (np.argmin(g, axis=1) )
    wordmap = wordmap.reshape(h,w)    
    return wordmap 
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----
def compute_dictionary_one_image(args):    
    i, alpha, path = args
    
    
        
# =============================================================================
#     for i in range(0,200):
#         alpha = np.random.randint(100,h*w - 100)
#    # if alpha in coll :
#     #    print("repeatttt")
#      #   coll.append(alpha)
#         temp  = new_filter[alpha]   
#    # print(temp)
#         temporary.append(temp)
#     temporary_array = np.asarray(temporary)
#     
#     print(temporary_array.shape)
#     
# =============================================================================    
    img = plt.imread("../data/" +path)
    img = img.astype('float')/255.0                               
    
    filter_responses = extract_filter_responses(img)
    #print (filter_responses.shape)
    #print (path)
    
    h,w,c = filter_responses.shape 
    new_filter = filter_responses.reshape((h*w),c) 
    #print (new_filter.shape)
    temporary_alpha_3F = []
    for j in range(0,alpha):
        rand_pixel = np.random.randint(0,h*w)
        #if alpha in coll :
           # print("repeatttt")
           # coll.append(alpha)
        temp  = new_filter[rand_pixel]
        temporary_alpha_3F.append(temp)
        temporary_alpha_3F_np = np.asarray(temporary_alpha_3F)    
    print(i)
    np.savez( "../code/" +str(i) + ".npz",temporary_alpha_3F_np)
    
    
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''
def compute_dictionary(num_workers):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

   # print ("innnn")
    np_temporary = np.empty((0,60))
    # ----- TODO -----    
    
    train_data = np.load("../data/train_data.npz")    
    alpha = 100
    [compute_dictionary_one_image((idx,alpha,path)) for idx,path in enumerate(train_data['files'])]  
# =============================================================================
#     argsss = [(idx,500,path) for idx,path in enumerate(train_data['files'])] 
#     "the code you want to test stays here"
#     with multiprocessing.Pool(os.cpu_count()) as p:
#         print("multiiii")    
#         
#     
#         p.map(compute_dictionary_one_image,(argsss ))  
# =============================================================================
   # print (args)
   # compute_dictionary_one_image(args)      
    for index in range(0,train_data['files'].shape[0]) :
        f = np.load("../code/" +str(index) + ".npz")
 
       # print (index , " : ",f['arr_0'].shape)
        np_temporary = np.append(np_temporary,f['arr_0'],axis = 0)
       # print (np_temporary.shape)       
    #print (np_temporary.shape)
    kmeans = sklearn.cluster.KMeans(n_clusters=140,n_jobs = -1).fit(np_temporary)
    dictionary = kmeans.cluster_centers_

    np.save("dictionary.npy",dictionary)


    