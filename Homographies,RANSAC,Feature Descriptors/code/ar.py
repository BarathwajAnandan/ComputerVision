import numpy as np
import cv2

from planarH import computeH
import  matplotlib.pyplot as plt


def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation
    '''

    #############################
    # TO DO ...
    H = H/H[2,2]
    h = np.matmul(np.linalg.inv(K), H)
    #print(h)
    
    h_12 = h[:, 0:2]
    U, S, V = np.linalg.svd(h_12, full_matrices=True)
    
   # print(U,S,V)

    m = np.array([[1, 0],[0, 1],[0, 0]])

    R12 = np.dot(np.dot(U, m), V)
    R3 = np.cross(R12[:, 0], R12[:, 1]).reshape(3, 1)
    R = np.append(R12, R3, axis=1)
    
    #print(R)
    
    detR = np.linalg.det(R) 
    if detR == -1:
        R[:, 2] *= -1

    
    lamdadash = np.sum(h_12/R12) / 6

    t = (h[:, 2] / lamdadash).reshape(3, 1)

    return R, t



def project_extrinsics(K, W, R, t):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        W - 3D planar points of textbook
        R - relative 3D rotation
        t - relative 3D translation
    OUTPUTS:
        X - computed projected points
    '''
    Rt = np.append(R, t, axis=1)
    #print(Rt)
    #W = np.stack(W,np.ones(len(W)))
    #print(W.shape)
    X = np.dot(np.dot(K, Rt), W)
    
    X = X[0:2, :] / X[2, :]
    #############################
    # TO DO ...

    return X


if __name__ == "__main__":
    # image
    im = cv2.imread('../data/prince_book.jpeg')
    
    
    W = np.array([[0.0, 18.2, 18.2, 0.0 ],[0.0, 0.0,26.0, 26.0],[0.0, 0.0,0.0,0.0 ]])

    X = np.array([[483, 1704, 2175, 67  ],[810, 781,  2217, 2286]])

    K = np.array([[3043.72, 0.0,1196.00 ],[0.0,3043.72,1604.00],[0.0,0.0,1.0]])

    #############################
    # TO DO ...
    # perform required operations and plot sphere
    
    
    sphere = open("../data/sphere.txt","r") 
    file = sphere.read()
    lines = file.split('\n')
    xcoor = lines[0].split('  ')  
    ycoor = lines[1].split('  ')
    zcoor = lines[2].split('  ')
    
    u = np.stack((xcoor,ycoor,zcoor,np.ones(len(xcoor))))
    u = u[:,1:]
    u = u.astype(np.float)
    W_xy = W[0:2, :]
    

    H = computeH(X, W_xy)
    
    Centre = np.array([825,1640,1])
    finalC = np.linalg.inv(H) @ Centre

    finalC = finalC/finalC[2]
   # print (finalC) 
    u[0]  +=  finalC[0]
    u[1] += finalC[1]    
    u[2] += -3.429
    
    R, t = compute_extrinsics(K, H)
    #print(R.shape)
 

    dots = project_extrinsics(K, u, R, t) 
    
    #print(u)
    
#    distanceToO = np.asarray([[310],[630]])
    
    SphereDots = dots 
    
    
    #print(SphereDots.shape)
    #print(SphereDots)
    im = plt.imread('../data/prince_book.jpeg')
    #fig, ax1 = plt.subplots(figsize=(16,12))
    
    #plt.imshow(im)
    #plt.plot(SphereDots[0, :], SphereDots[1, :], 'y-', markersize=1)
    #plt.imsave("../results/AR.jpg",im)
    #plt.show()