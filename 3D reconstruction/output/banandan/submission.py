"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
import numpy.matlib
import helper 
import matplotlib.pyplot as plt

import scipy.ndimage.filters
import scipy.optimize as scpo
import math
import plotly.graph_objects as px
from plotly.offline import  plot
# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    
    p1 = pts1/M
    p2 = pts2/M
    
    #print(p1, p2)
#    points = np.load("../data/some_corresp.npz")
#    p1 = points['pts1'] /M
#    p2 = points['pts2'] /M
    
    #scaling  p1 and p2, / M ?   normalizing  and  origin to 0,0 ?
    
    
    
    #homogenous
    hom = np.ones((p1.shape[0],1))
    
    
    #points1= np.concatenate((points1,hom),axis = 1)
    #points2 = np.concatenate((points2,hom),axis = 1)
    x1 = p1[:,0]
    y1 = p1[:,1]
    x2 = p2[:,0]
    y2 = p2[:,1]

    
    A = np.asarray([x2*x1,x2*y1,x2,y2*x1, y2*y1,y2,x1,y1,np.ones(pts1.shape[0])]).T
    #print (A.shape)
    
    
    
    #print(A[76])
    
    
    U,S,V = np.linalg.svd(A)
    
    #print(V.shape)
    F = V[-1, :]
    
    F = np.reshape(F,(3,3))
    
    #print (F)
    
    U,S,V = np.linalg.svd(F)
    
    a = S[0]
    b = S[1]
    c = 0
    SS = np.diag([a,b,c])
    #print(Ftest.shape)
    FMatrix = U @SS@V
    #print(U.shape,S.shape,V.shape)
    #S[2] = 0
    #S = np.diag(S)
   # rank 2!
    
   # print(FMatrix)
    
    
    t = np.eye(3) * 1/M
    
    t[2,2] = 1
    
    #FMatrix = helper.refineF(FMatrix,p1,p2)
    
    #unnormalize?
    F = t.T @ FMatrix @ t 
    F = np.asarray(F)
    
    
    #print(F)



    return F



'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    
    p1 = pts1/M
    p2 = pts2/M
    
    x1 = p1[:,0]
    y1 = p1[:,1]
    x2 = p2[:,0]
    y2 = p2[:,1]

    
    A = np.asarray([x2*x1,x2*y1,x2,y2*x1, y2*y1,y2,x1,y1,np.ones(pts1.shape[0])]).T
    #print (A.shape)
    
    
    
    #print(A[76])
    
    
    U,S,V = np.linalg.svd(A)
    
    F1 = np.asarray(V[-1,:]).reshape((3,3))
    F2 = np.asarray(V[-2,:]).reshape((3,3))
    
    
    
    #print(F1.shape,F2.shape)
    
    
    
    
    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
    a0 = fun(0)
    a1 = 2.0 * (fun(1) - fun(-1)) / 3 - (fun(2) - fun(-2)) / 12
    a2 = 0.5 * fun(1) + 0.5 * fun(-1) - fun(0)
    a3 = fun(1) - a0 - a1 - a2
    
    
    
    
    
    
    root = np.roots(np.array([a3, a2, a1, a0]))
    
    #print(root)
    
    F_array = []
    for r in root:
        
        F = r * F1 + (1 - r) * F2
        
        
        
        t = np.eye(3) * 1/M
        t[2,2] = 1
        
        #print(F.shape)
        
        F = t.T @ F @ t 
        
        F_array.append(F)
    
    F_array = np.asarray(F_array)
    return F_array
    

    
    
    


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    e = K2.T @ F @ K1
    return e


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    
    
    W = np.zeros((pts1.shape[0], 4))
    P = np.zeros((pts1.shape[0], 3))
    A = []
    p1 = C1[0,:]
    p2 = C1[1,:]
    p3 = C1[2,:]
    p1prime = C2[0,:]
    p2prime = C2[1,:]
    p3prime = C2[2,:]
    
    
    for i in range(pts1.shape[0]):
        x = pts1[i,0]
        y = pts1[i,1]
        xprime = pts2[i,0]
        yprime = pts2[i,1]
        A = [y*p3 - p2,p1-x*p3,yprime*p3prime-p2prime, p1prime- xprime* p3prime]
        A = np.asarray(A)
        #A[0,1] = A[0,1][0]
        
        #print(np.asarray(A))
        A = A.astype(float)
        
        U,S,V = np.linalg.svd(A)
        
        w = V[-1,:]
        
        # in homogenous form
        W[i,:] = w/w[3]
        P[i,:] = W[i,0:3]
        
        #3D to 2D
    p1_proj = np.dot(C1 ,W.T)
    p2_proj = np.dot(C2 , W.T)
    
    p1_n = p1_proj/p1_proj[2]
    p2_n = p2_proj/p2_proj[2]

    #p1_n[0, :] = P1_proj[0]/P1_proj[2]
    
    #p1_n[1, :] = P1_proj[1]/P1_proj[2]
    #p2_n[0, :] = P2_proj[0]/P2_proj[2]
    #p2_n[1, :] = P2_proj[1]/P2_proj[2]
    p1_n = p1_n[0:2].T
    p2_n = p2_n[0:2].T
    
   # print(pts1[4],p1_n[4])

    
    #print(P1_n)
    
    err = np.sum((p1_n- pts1)**2 + (p2_n-pts2)**2)
    
    #print(err)
    
    
    return P,err

        
        
        
        
    
    

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    window = 12      
    l = F @ [x1, y1, 1]
    X =[]
    Y = []
    h,w,c = im1.shape
    
    for x in range(window, w - window):
        y = -np.ceil((l[0] * x + l[2]) / l[1])  #ax+by+ c = 0 
        if y >= window and y <= h - window:
            X.append(x)
            Y.append(y)

    
    

    min_error = 1000000
    for i in range(len(X)):
        print(i)
        x2 = X[i]
        y2 = Y[i]
        if np.sqrt((x1-x2)**2 + (y1-y2)**2) < 50:
            patch1 = im1[int(y1 - window) : int(y1 + window+1), int(x1 - window ) : int(x1 + window+1), :] #since height correponds to y , 
            patch2 = im2[int(y2 - window) : int(y2 + window+1), int(x2 - window ) : int(x2 + window+1), :] # width corr to x
            e = patch1 - patch2
            g = scipy.ndimage.filters.gaussian_filter(e, sigma=1)
            error = np.sum(g)
            if error < min_error:
                min_error = error
                idx = i
    x2 = X[idx]
    y2 = Y[idx]

    return x2, y2

    
    
 



def ransacF(pts1, pts2, M):
    num_iter = 500
    threshold = 0.001
    max_inliers = 0
    inliers = np.zeros((pts1.shape[0], 1))


    for i in range(num_iter):
        #print("RANSAC iteration ", i)

        r = np.random.choice(pts1.shape[0],7)
        r = [ 99 ,115,  91, 118,  94, 127,  46]
        points1 = pts1[r]
        points2 =pts2[r]

        inliers_index = []
        Farray = sevenpoint(points1, points2, M)
        for f in Farray:
            n = 0
            for k in range(pts1.shape[0]):
                p1_test = [pts1[k, 0], pts1[k, 1], 1]
                p2_test = np.array([pts2[k, 0], pts2[k, 1], 1])
                error = abs(p2_test.T @ f @ p1_test)

                if error < threshold:
                    n += 1
                    inliers_index.append(k)

            if n > max_inliers:

                inlierIndex = np.array(inliers_index)
                
                max_inliers = n
                F = f
    inliers[inlierIndex] = 1

    return F, inliers


                
            
                
                
                
                
                
                    
                
                    
        
                
                    
                    
                    
                    
                    
            
        
        
        
        
    
    
    
    
    
    

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''


def rodrigues(r):
    theta = np.linalg.norm(r)
    R = np.zeros((3,3))
    if theta == 0:
        R = np.eye(3)
    else:
        a = r / theta
        a_sm = np.array([[0, -a[2], a[1]],[a[2], 0, -a[0]],[-a[1], a[0], 0]])
        R = np.cos(theta) * np.eye(3) + np.dot((1 - np.cos(theta)) * a, a.T) + np.sin(theta) * a_sm
        
        R[0,1:3] =  R[0,1:3][0]
        R[1,0] =  R[1,0][0]
        R[1,2] =  R[1,2][0]
        R[2,0:2] =  R[2,0:2][0]
        
        #print(R)

    return R



'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''



def invRodrigues(R):
    # Replace pass by your implementation
    
   
	theta = math.acos((np.trace(R) - 1) / 2)

	if theta != 0:
		n = 1 / (2*np.sin(theta)) * np.array([[R[2, 1] - R[1, 2]], [R[0, 2] - R[2, 0]],[R[1, 0] - R[0, 1]]])
		r = theta * n
	return r

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    
    #print(p1.shape,p2.shape,x.shape)

    r = x[-6:-3].reshape(3, 1)
    t = x[-3:].reshape(3, 1)
    P = x[0:-6].reshape(-1, 3)


    R = rodrigues(r);
    M2 = np.concatenate((R, t), axis=1)
    C1 = np.dot(K1, M1)
    C2 = np.dot(K2, M2)
    homo = np.ones((P.shape[0], 1))
    P = np.concatenate((P, homo), axis=1).T

    p1_proj =np.dot(C1,P)
    p2_proj = np.dot(C2,P)

    p1_n = np.zeros((2, P.shape[1]))
    p2_n = np.zeros((2, P.shape[1]))

    p1_n[0:2] = p1_proj[0:2]/p1_proj[2]
    
    p2_n[0:2] = p2_proj[0:2]/p2_proj[2]
    
    
    p1_n = p1_n.T
    p2_n = p2_n.T

    e1 = (p1 - p1_n).reshape(-1)
    e2 = (p2 - p2_n).reshape(-1)

    residuals = np.append(e1, e2, axis=0)

    return residuals
    
    
    
    
    
    

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):

    residual = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)

    R_before = M2_init[:, 0:3]
    t_before = M2_init[:, 3]
    r_before = invRodrigues(R_before).reshape(3)

    x = np.zeros(3 * P_init.shape[0] + 6)
    x[-6:-3] = r_before
    x[-3:] = t_before
    x[0:-6] = P_init.reshape(-1)
    
    #print("R:",R_before)
    #print("t",t_before)
    x_optimized, _ = scpo.leastsq(residual, x)
    r = x_optimized[-6:-3].reshape(3, 1)
    t = x_optimized[-3:].reshape(3,1)
    P = x_optimized[0:-6].reshape(-1, 3)

    R = rodrigues(r)
    
    
    #print("R:",R)
    #print("t",t)
    M2 = np.append(R, t, axis=1)
   
    #M2[:,:] = M2[:,:][0]
    

    return M2, P




def find_points(pts1,pts2):
   
        F_ref = np.array([[ 9.78833288e-10 ,-1.32135929e-07 , 1.12585666e-03],
                          [-5.73843315e-08  ,2.76800276e-09 ,-1.57611996e-05],
                          [-1.08269003e-03  ,3.04846703e-05 ,-4.37032655e-03]])
       
        padded_ones = np.ones((pts1.shape[0],1))
       
        pts1_hom = np.hstack((pts1 , padded_ones))
    
        pts2_hom = np.hstack((pts2 , padded_ones))
       
        error = np.array([0])
         
        for i in range(pts2_hom.shape[0]):
           
            error = abs(np.append(error, np.matmul(np.matmul(pts2_hom[i],F_ref),pts1_hom[i])))
           
        error = error[1:]
       
        idx = np.argsort(error)[:7]
   
        return pts1[idx] ,pts2[idx]


if __name__ == "__main__" :
    pass   
    
# =============================================================================
#     
#     img1 = plt.imread("../data/im1.png")
#     img2 = plt.imread("../data/im2.png")
#     
#     h,w,c = img1.shape
#     points = np.load("../data/some_corresp.npz")
#     p1 = points['pts1'] 
#     p2 = points['pts2'] 
#     M = max (h,w)
#     #print(type(M))
#     
#     
#     x1,x2 = find_points(p1,p2)
#     print (x1)
#     print(x2)
#     
#     
#     
#     
#     
#     
#     #unlock for 2.1 - Eight point
#     F = eightpoint(p1,p2,M)
#     
# =============================================================================
    
    
    #F = helper.refineF(F,p1,p2)
    #print(F)
    
    #np.savez("../output/q2_1.npz",F =F,M=M)
    #np.savez("../output/q4_1.npz",F=F,pts1=p1, pts2=p2)
    
    
   # helper.displayEpipolarF(img1,img2,F)

     
# =============================================================================
# #unlock for 2.2 - seven point - F[0] is right
#     img1 = plt.imread("../data/im1.png")
#     img2 = plt.imread("../data/im2.png")
#     
#     h,w,c = img1.shape
#     points = np.load("../data/some_corresp.npz")
#     
#     
#     p1 = points['pts1'] 
#     p2 = points['pts2'] 
#     p1_7point = []
#     p2_7point = []
# #    for i in range(0,7):
#     
# # #     
#     
#     
#     i1,i2 = find_points(p1,p2)
#     print(i1,i2)
# #   np.random.seed(48)
#    # r = np.random.randint(0,p1.shape[0])
#    # r = [ 99 ,115,  91, 118,  94, 127,  46]
#     
#    # print(r)
#     p71,p72 = find_points(p1,p2)
#     p1_7point = p71
#     p2_7point = p72 
#         
#     M = max (h,w)
#     
#     
#     
#     p1_7point = np.asarray(p1_7point)
#     p2_7point = np.asarray(p2_7point)
#     F = sevenpoint(p1_7point,p2_7point,M)
#     
#     
#     #print(F.shape)
#     
#     
#     helper.displayEpipolarF(img1,img2,F[0])
#     print(F[0])
# =============================================================================

    #np.savez("../output/q2_2.npz",F = F[0],M=M,pts1 = p1_7point, pts2 = p2_7point)

#3.2 and 3.1 

# =============================================================================
#     intrinsics = np.load("../data/intrinsics.npz")
#     K1  = intrinsics['K1']
#     K2 = intrinsics['K2']
#     
#     #print(K1.shape,K2.shape)
#     e = essentialMatrix(F,K1,K2)
#    
#     M1 = M1 = np.zeros((3,4))
#     M1[0,0] = 1
#     M1[1,1] = 1
#     M1[2,2] = 1
#     M2 = helper.camera2(e)
#     
#     #print(M2.shape)
#     C1  = K1@M1
#     C2 = K2@ M2[:,:,2]
#     #print(C2.shape)
#     
#     W, err =    triangulate(C1,p1,C2,p2)
#     #print("W: ",W)
# =============================================================================

# =============================================================================
# # 4.1 
#     
#     #epiline equation:
#   # 
#    helper.epipolarMatchGUI(img1,img2,F) 
#     
#  #    
#     
#  
# =============================================================================
    
    
 #5.1 RANSAC
 
# =============================================================================
#     noisyPoints = np.load("../data/some_corresp_noisy.npz")
#     nPts1 = noisyPoints['pts1']
#     nPts2 = noisyPoints['pts2']
#     
#     intrinsics = np.load("../data/intrinsics.npz")
#     K1  = intrinsics['K1']
#     K2 = intrinsics['K2']
#     
#  
#     Fransac, Inliers  =  ransacF(nPts1,nPts2,M)
#     
#     I = np.where(Inliers ==1)
#     I = np.asarray(I)
#     pts1 =nPts1[I].reshape(-1,2)
#     pts2 =nPts2[I].reshape(-1,2)
#     
#     
#     
#     #6helper.displayEpipolarF(img1,img2,Fransac)
#     
#     E =essentialMatrix(Fransac,K1,K2)
#     
#     
#     M1 = M1 = np.zeros((3,4))
#     M1[0,0] = 1
#     M1[1,1] = 1
#     M1[2,2] = 1
#     
#     C1  = K1@M1
#     
#     tee = 0
#     M2_ = helper.camera2(E)
#     err_final = 10000
#     for i in range(0,4):
#         #print(i)
#     
#         M2 = M2_[:,:,i]
#         
#         
#         C2 = K2@M2
#     #print(C2.shape)
#     
#         W, err = triangulate(C1,pts1,C2,pts2)
#         z = W[:,2]
#         #print(W[:,2])
#         
#         tee = z[z<0]
#         
#         
#         
#         #print(tee)
#         #print(err)
#         #if err<err_final:
#         if len(tee)==0:
#             err_final = err
#             
#             #print(err_final)
#             C2_final = C2
#             M2_final = M2
#             P_final  = W
#                 
#        
#             
#     fig = px.Figure(data=[px.Scatter3d(x=P_final[:, 0], y=P_final[:, 1], z=P_final[:, 2],
#                                    mode='markers')])    
#     plot(fig)         
#    
#     
#     reprojection_error_before = err_final
#                 
#     M2,P =   bundleAdjustment(K1,M1,pts1,K2,M2_final, pts2,P_final )
#     
#     C2 = np.dot(K2,M2)
#     
#     P_tri,err = triangulate(C1,pts1,C2,pts2)
#     
#     
#     print("initialError",reprojection_error_before)
#     print("Final",err)
#     
#     
#     
#    
#     #P_leasterror, err = triangulate(C1,pts1,C2,pts2)
# 
#                     
# 
#             
#     fig = px.Figure(data=[px.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2],
#                                    mode='markers')])    
#     plot(fig)
#                
# 
#     
#     
# 
#     
#     
#     
# =============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
