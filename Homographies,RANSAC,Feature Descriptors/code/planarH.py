import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                coordinates between two images
    OUTPUTS
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
                equation
    '''    
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    N = p2.shape[1] 
    A = np.zeros((N*2,9))  # 9 because 9 elements in H Matrix ? and each point = 2 eq .  
    # elements into A matrix . two eq for one set of points.
    x = p1[0,:]   #values of x in p2
    y = p1[1,:]   #array of y in p2
    u = p2[0,:]
    v = p2[1,:]
    for i in range(N):
        A[(2*i)+1] = [-u[i],-v[i],-1,0,0,0,x[i]*u[i]  ,x[i]*v[i], x[i]]
        A[2*i] = [0,0,0,-u[i],-v[i],-1, y[i]*u[i], y[i]*v[i], y[i]]
    U,S,V = np.linalg.svd(A)    
    H = V[-1,:]
    H = np.reshape(H,(3,3))
    H2to1 = H
    #print(H)
    return H2to1
def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using RANSAC
    
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches         - matrix specifying matches between these two sets of point locations
        nIter           - number of iterations to run RANSAC
        tol             - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''
    l = len(matches)
    max_Inliers = -1  
    img1MatchIdx = matches[:,0]
    img2MatchIdx = matches[:,1]
    X_homogen = np.append(locs1[img1MatchIdx,0:2].T,np.ones((1,l)), axis  = 0)
    U_homogen = np.append(locs2[img2MatchIdx, 0:2].T,np.ones((1,l)), axis = 0)

    for iteration in range(num_iter):
        randomness = np.random.choice(l,4,replace =False)    
        p1 = locs1[ img1MatchIdx[randomness] ,0:2 ].T
        p2 = locs2[ img2MatchIdx[randomness] ,0:2 ].T          
        #print(p1.shape)
        H = computeH(p1,p2)
        X_Obt = np.matmul(H, U_homogen)
        Z = X_Obt[2,:]        
        X_Obt =X_Obt/Z
        E  = X_homogen- X_Obt
        D = (sum(E**2))
        N_Inliers = len(np.argwhere(D <=tol*2))
        if N_Inliers > max_Inliers:
            
            max_Inliers = N_Inliers
            max_Inliers_H = H         
    print("RANSAC max number of inliers: ", max_Inliers)
    return max_Inliers_H

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)