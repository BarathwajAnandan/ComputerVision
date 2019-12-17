import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform as af

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    h,w = It1.shape
    h1,w1 = It.shape
    
    current_x = np.linspace(0,h-1,h)
    current_y = np.linspace(0,w-1,w)
    It_x = np.linspace(0,h1-1,h1)
    It_y = np.linspace(0,w1-1,w1)
    
    x,y = np.meshgrid(current_x,current_y)
    
    #print("x:",x.shape,"y:",y.shape)
  #  ev_It_orig = RectBivariateSpline(It_x,It_y,It)
    
    #mask = np.ones((1,h*w))
    
    GradientY,GradientX = np.gradient(It)
    

    
    #NablaI
    warped_Gx = af(GradientX,M).flatten() 
    warped_Gy = af(GradientY,M).flatten()

    
    x = x.flatten()
    y = y.flatten()
    #print (GradientX,"breakkkkk",warped_Gx)
    A = [warped_Gx*x,warped_Gx*y,warped_Gx*1, warped_Gy*x,warped_Gy*y,warped_Gy*1]
    A = np.asarray(A).squeeze().T


    A_ = np.linalg.inv([(A.T)@A ]) @ A.T
    
    #print(A_)
    
    #print(A_)
    
    
    
    
    thresold = 1  
    delP_norm = 5
    while   delP_norm> thresold:

        
 #warping It and not It1 for Inverse
        warped_It1 = af(It1,M)
        warped_It1 = warped_It1.flatten()
        #warped_mask = af(mask,M)
       # warped_mask = warped_mask.squeeze()
        It = It.flatten()
        #print(It1.shape , warped_mask.shape)
        #warped_It1 = It1.flatten() 
        
        #print(warped_Current.shape,warped_It1.shape)
        #modified A and b matrix
        
        b = It -  warped_It1  
        
        b = b.flatten()
        
        
     #least Sq replaced by the formula for DelP
     
        delP = A_ @ b
        delP_norm = np.linalg.norm(delP)
        delM = np.asarray([[1+delP[0,0], delP[0,1],delP[0,2]],[delP[0,3],1+delP[0,4],delP[0,5]],[0,0,1]])
        delMInv = np.linalg.inv(delM)            
        M = M @ delMInv 
        
        #print (M)# p dot delP
    return M
