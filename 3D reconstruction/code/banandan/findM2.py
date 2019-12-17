import numpy as np
import matplotlib.pyplot as plt
import submission
import helper


'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def test_M2_solution(pts1, pts2, intrinsics, M):
    '''
    Estimate all possible M2 and return the correct M2 and 3D points P
    :param pred_pts1:
    :param pred_pts2:
    :param intrinsics:
    :param M: a scalar parameter computed as max (imwidth, imheight)
    :return: M2, the extrinsics of camera 2
    		 C2, the 3x4 camera matrix
    		 P, 3D points after triangulation (Nx3)
    '''
    F = submission.eightpoint(pts1,pts2,M)
    F = np.asarray(F)
    
    intrinsics = np.load("../data/intrinsics.npz")
    K1  = intrinsics['K1']
    K2 = intrinsics['K2']
    
    #print(K1.shape,K2.shape)
    e = submission.essentialMatrix(F,K1,K2)
   
    M1 = M1 = np.zeros((3,4))
    M1[0,0] = 1
    M1[1,1] = 1
    M1[2,2] = 1
    M2_ = helper.camera2(e)
    
    
    print(M2_.shape)
    
    
    
    C1  = K1@M1
    tee = 0
    err_final = 10000
    for i in range(0,4):
        print(i)
    
        M2 = M2_[:,:,i]
        
        
        C2 = K2@M2
    #print(C2.shape)
    
        W, err =    submission.triangulate(C1,pts1,C2,pts2)
        z = W[:,2]
        #print(W[:,2])
        
        tee = z[z<0]
        
        
        
        #print(tee)
        #print(err)
        #if err<err_final:
        if len(tee)==0:
            err_final = err
            
            #print(err_final)
            C2_final = C2
            M2_final = M2
            P_final  = W
            print(P_final)
            print(err_final)

    
    
	

    return M2_final, C2_final, P_final


if __name__ == '__main__':
    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    intrinsics = np.load('../data/intrinsics.npz')
    
    img1 = plt.imread("../data/im1.png")
    
    h,w,c = img1.shape
    M = max (h,w)

    M2, C2, P = test_M2_solution(pts1, pts2, intrinsics,M)
    
    #print("P:", P)
    #np.savez('../output/q3_3.npz', M2=M2, C2=C2, P=P)
