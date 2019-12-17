import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    
  
    h,w,c = im1.shape[0:3]
    h2,w2,c2 = im2.shape[0:3]
    pano_im = cv2.warpPerspective(im2, H2to1, (1700,700))
   #pano_im[:,:,k] = np.max(im1[:,:,k], pano_im[:,:,k])
   
    cv2.imwrite('../results/6_1.1.jpg', pano_im)
    
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            pano_im[i,j,:] = np.maximum(im1[i,j,:], pano_im[i,j,:])
           # print (pano_im)
    
    

    

    return pano_im
def imageStitching_noClip(im1, im2, H2to1):
    
    
    
    
    '''
    Returns a panorama of im1 and im2 using the given
    homography matrix without cliping.
    '''
    ######################################
    # TO DO ...
    #pano_im = cv2.warpPerspective(im2, H2to1, (2500, 2500))
    # print("im1 shape: ", im1.shape)
    # print("im2 shape: ", im2.shape)
    #pano_im = cv2.warpPerspective(im2, H2to1, (2500, 2500))
    #clockwise direction
    w,h = im2.shape[0:2]
    w1,h1 = im1.shape[0:2]
    
    corners = [[0,h-1,h-1,0], [0,0,w-1,w-1]]
    
    ones = [[1,1,1,1]]
    
    U = np.vstack((corners,ones))
    
    X = np.dot(H2to1,U)
    
    Z = X[2,:]
    X = X/Z
    
    
    X_max  = max(np.max(X[0]), w1-1)
    X_min = min(np.min(X[0]),0)
    Y_max = max(np.max(X[1]), h1-1)
    Y_min = min(np.min(X[1]),0)
    
    
    outsize = (int(X_max - X_min), int(Y_max-Y_min) )
    
    
    
    M_scale = np.identity(3)
    
    M_trans = np.array([[1, 0, 0], [0, 1, -Y_min], [0, 0, 1]])
    
    M =  np.dot(M_trans, M_scale)   

    warp_im1 = cv2.warpPerspective(im1, M, outsize)
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), outsize)
    
    
    
    pano_im = np.maximum(warp_im1,warp_im2)
    

    
    
    
    
    
    #print (U.shape)
    
    
    

    return pano_im
def generatePanorama(im1, im2):
    
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    
    Pano = imageStitching_noClip(im1, im2, H2to1)

    return Pano
if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    np.save("../results/q6_1.npy",H2to1)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    im6_1 = imageStitching(im1, im2, H2to1)
    
    #cv2.imshow("hey",im6_1)
    im6_2 = imageStitching_noClip(im1, im2, H2to1)
    im6_3 = generatePanorama(im1, im2)
   
    cv2.imwrite('../results/6_1.jpg', im6_1)
    cv2.imwrite('../results/6_2.jpg', im6_2)
    cv2.imwrite('../results/6_3.jpg', im6_3)
    
    
    
   # cv2.waitKey(0)
    #cv2.destroyAllWindows() 
