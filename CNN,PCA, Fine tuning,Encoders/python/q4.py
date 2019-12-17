import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt
import matplotlib.patches as patchess


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    
    # processing needs gray, opening etc.
    gray = skimage.color.rgb2gray(image)
   
    
    #noise reduction
    blurred = skimage.filters.gaussian(gray, sigma=1.0)
    #plt.imshow(blurred)
    
    #getting thresold value based on OTSU?
    threshold = skimage.filters.threshold_otsu(blurred)
    
    #around 0.46?   
    #get value after thresolding
    binary = gray <= threshold
    
    
    #erosion followed by dilation
    opening = skimage.morphology.binary_opening(binary)
    
    
    #getting area based on connectivity with neigbouring pixels ? probably means that if the pixel 
    #vlaues are same , it joins em ? since binary is only grayscale values, it's easier for grping?
    
    labels = skimage.measure.label(opening)
    regions = skimage.measure.regionprops(labels)
    
    Region = 0
    for i in regions :
        Region = Region + i.area
    MeanRegion = Region/len(regions)
    
    
    
    for i in regions :
        if i.area > MeanRegion/2:
            
            bboxes.append(i.bbox)
            x1,y1,x2,y2=i.bbox
        
            #Plot check if bboxes are  right
            #plt.gca().add_patch(patchess.Rectangle((y1,x1), y2-y1,x2-x1 ,fill=False, edgecolor='red', linewidth=1))
            #plt.imshow(opening)
            
            #plt.set_axis_off()
            #plt.tight_layout()
    plt.show()  

    
    
    
    
    
    
    
    
    bw  = 1 - opening
    
    bw = np.float64(bw)
    
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################

    return bboxes, bw


