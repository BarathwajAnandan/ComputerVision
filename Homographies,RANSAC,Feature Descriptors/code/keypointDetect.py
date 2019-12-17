import numpy as np
import cv2
import matplotlib.pyplot as plt

PRINT = 0
def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)

    
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# =============================================================================
#     cv2.imshow('Pyramid of image', im_pyramid)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# =============================================================================


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    
    h,w,l = gaussian_pyramid.shape

    
    gaussian_pyramid = np.split(gaussian_pyramid, gaussian_pyramid.shape[2], axis=2)

    
    DoG_pyramid = np.ndarray((h,w,0))
    #print("dog shape:",dog.shape)
    for i in range(1,len(gaussian_pyramid)):
        a = gaussian_pyramid[i]
        
        b = gaussian_pyramid[i-1]
        #print(a.shape,b.shape)
        c  = a-b
        

        
        #print ('c:', c.shape)
        DoG_pyramid = np.append(DoG_pyramid,c , axis = 2)
        DoG_pyramid = np.asarray(DoG_pyramid)

    '''
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    '''

    
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    h,w,l = DoG_pyramid.shape

    principal_curvature = np.ndarray((h,w,0))

    DoG_pyramid = np.split(DoG_pyramid, DoG_pyramid.shape[2], axis=2)

    
    for i in range(0,len(DoG_pyramid)):
        DoG_pyramid_level = DoG_pyramid[i]

        DoG_pyramid_x = cv2.Sobel(DoG_pyramid_level,ddepth = -1  ,dx = 1,dy=0)
        
        DoG_pyramid_y = cv2.Sobel(DoG_pyramid_level,ddepth = -1,dx = 0,dy = 1)

        H_xy  =  cv2.Sobel(DoG_pyramid_x,ddepth = -1,dx = 0,dy =1)
        H_yx =  cv2.Sobel(DoG_pyramid_y,ddepth = -1,dx = 1,dy =0)
        H_xx =  cv2.Sobel(DoG_pyramid_x,ddepth = -1,dx =1,dy=0)
        H_yy =  cv2.Sobel(DoG_pyramid_y,ddepth = -1,dx =0,dy =1)
    
        Trace = H_xx + H_yy
        Trace_sq = Trace**2
        Det =  H_xx*H_yy - H_xy*H_yx
 
        R = Trace_sq/Det

        R = R[: ,:,np.newaxis]

        principal_curvature = np.append(principal_curvature,R,axis = 2)

    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = np.zeros((1,3) , dtype = int )
    h,w,l = DoG_pyramid.shape
    

    for L in range(l):      
        for  H in range(1,h-1):
            for W in range(1,w-1):
                if DoG_pyramid[H,W,L] >= th_contrast and principal_curvature[H,W,L] <= th_r :
                    NN = DoG_pyramid[H-1:H+2,W-1:W+2,L]

                    if L == 0: 
                        NN = np.append(NN,DoG_pyramid[H,W,1])
                    elif L == 4:
                        NN = np.append(NN,DoG_pyramid[H,W,3])
                    else:                      
                        NN = np.append(NN,DoG_pyramid[H,W,L-1])
                        NN = np.append(NN,DoG_pyramid[H,W,L+1])
                    Ex_min_idx = np.argmin(NN)
                    Ex_max_idx = np.argmax(NN)

                    if Ex_min_idx == 4 or Ex_max_idx == 4:
                        extrema = (W,H,L)

                        extrema = np.reshape(extrema,(1,3))

                        locsDoG  = np.append(locsDoG,extrema , axis = 0)
                    
    
    ##############
    #  TO DO ...
    # Compute locsDoG here
    #print(locsDoG.shape)
    locsDoG = locsDoG[1:]
    return locsDoG
  

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    
    ##########################
    # TO DO ....
    gaussian_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gaussian_pyramid, levels)
    p_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, p_curvature, th_contrast, th_r)
    
    
    
    return locsDoG, gaussian_pyramid


if __name__ == '__main__':

    levels = [-1,0,1,2,3,4]
    img = plt.imread('../data/model_chickenbroth.jpg')
    
    
    GPyramid = createGaussianPyramid(img)
    displayPyramid(GPyramid)
    

    DoG, DoG_levels = createDoGPyramid(GPyramid, levels)
    displayPyramid(DoG)
    

    pc_curvature = computePrincipalCurvature(DoG)
    

    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG, DoG_levels, pc_curvature, th_contrast, th_r)
    #print(locsDoG.shape)

    for i in range(len(locsDoG)):
        
        row = locsDoG[i][0]
        col  = locsDoG[i][1]
        cv2.circle(img, (row, col), 1, color=(0,255,0) , lineType = cv2.LINE_4)
        #cv2.namedWindow("image",cv2.WINDOW_FREERATIO)
    if PRINT == 1:    
        cv2.imshow('image', img)
        cv2.imwrite('../results/keypoints.jpg', img)
        plt.imshow(img)
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

