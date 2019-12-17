import os
import numpy as np
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

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    #plt.imshow(bw)
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
        letter = bw[y1:y2+1, x1:x2+1]
        

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
        letter = np.pad(letter,((20,20),(15,15)),'maximum')
        letter = skimage.morphology.binary_erosion(letter)
        letter = skimage.filters.gaussian(letter,sigma = 1)
        #letter = skimage.filters.median(letter)
        letter = skimage.transform.resize(letter,(32,32)).T
        #letter = letter.reshape((32,32))
        #letter = letter.resize((32,32))
        #plt.imshow(letter)
        plt.show()
    
        letters.append(letter.reshape(-1))
    letters = np.asarray(letters)
    
    #print(letters.shape)


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    # letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    h1 = forward(letters, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)

    # ground_truth = np.argmax(test_y, axis=1)
    predicted = np.argmax(probs, axis=1)
    # print(predicted)
    letter_list = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    predicted_letters = letter_list[predicted]

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


