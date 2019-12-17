import numpy as np
from scipy.ndimage import affine_transform as af
from LucasKanadeAffine import LucasKanadeAffine
import skimage.morphology 
from InverseCompositionAffine import InverseCompositionAffine




def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here\
    key= 1
    #0 is for forward
    #1 is for Inverse approach
    if (key == 0):
        
        clip = np.zeros(image1.shape, dtype=np.float32)
        clip[10:170, 55:210] = 1.0
        M = LucasKanadeAffine(image1,image2)
        #M = InverseCompositionAffine(image1,image2) 
        mask = np.zeros(image1.shape, dtype=bool)
        warped_Image1 = af(image1,M)
        mask = abs(warped_Image1-image2)  *clip
        #print(mask.shape)
        mask = mask>0.33
        #print(mask.shape)
        #mask = skimage.morphology.binary_erosion(mask)
        mask = skimage.morphology.binary_dilation(mask, np.ones((8, 8)))
        
    else:
        clip = np.zeros(image1.shape, dtype=np.float32)
        clip[20:170, 55:210] = 1.0
        #M = LucasKanadeAffine(image1,image2)
        M = InverseCompositionAffine(image1,image2) 
        mask = np.zeros(image1.shape, dtype=bool)
        warped_Image1 = af(image1,M)
        mask = abs(warped_Image1-image2)  *clip
        #print(mask.shape)
        mask = mask>0.3
        #print(mask.shape)
        #mask = skimage.morphology.binary_erosion(mask)

        mask = skimage.morphology.binary_dilation(mask, np.ones((5, 5)))
        
    
    
    
    
    #print (M)
    
        
    
    #mask = skimage.morphology.binary_dilation(mask,np.ones((9,9)))
    
    return mask


