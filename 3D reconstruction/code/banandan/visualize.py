'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''


from plotly.offline import  plot
import matplotlib.pyplot as plt
import numpy as np
import submission 
import helper
import plotly.graph_objects as px


img1 = plt.imread("../data/im1.png")
img2 = plt.imread("../data/im2.png")
points = np.load("../data/some_corresp.npz")
intrinsics = np.load("../data/intrinsics.npz")
K1  = intrinsics['K1']
K2 = intrinsics['K2']
p1 = points['pts1'] 
p2 = points['pts2'] 
h,w,c = img1.shape

pointsTemple = np.load("../data/templeCoords.npz")
xT1 = pointsTemple['x1']
yT1 = pointsTemple['y1']
xT2 = []
yT2 = []
M = max (h,w)


F = submission.eightpoint(p1,p2,M)
E = submission.essentialMatrix(F,K1,K2)



for  i in range (len(xT1)) :        
    
    #print(xT1[0])
    x2,y2 = submission.epipolarCorrespondence(img1,img2,F,xT1[i,0],yT1[i,0])
    xT2.append(x2)
    yT2.append(y2)
    
points = [xT2,yT2]
#print(xT2,yT2)


ptsT1 = np.stack((xT1[:,0],yT1[:,0]))
ptsT1 = np.moveaxis(ptsT1,0,-1)


ptsT2= np.stack((xT2,yT2))
ptsT2 = np.moveaxis(ptsT2,0,-1)

    


M1 = M1 = np.zeros((3,4))
M1[0,0] = 1
M1[1,1] = 1
M1[2,2] = 1
C1  = K1@M1


tee = 0
err_final = 10000
M2_ = helper.camera2(E)
for i in range(0,4):
    print(i)

    M2 = M2_[:,:,i]
    
    
    C2 = K2@M2
#print(C2.shape)

    W, err = submission.triangulate(C1,ptsT1,C2,ptsT2)
    z = W[:,2]
    #print(W[:,2])
    
    tee = z[z<0]
    
    
    
    #print(tee)
    #print(err)
    if err<err_final:
        if len(tee)==0:
            err_final = err
            
            #print(err_final)
            C2_final = C2
            M2_final = M2
            P_final  = W
            print(P_final)
            print(err_final)
            
 

#print("plottt")

#fig = px.Figure(data=[px.Scatter3d(x=P_final[:, 0], y=P_final[:, 1], z=P_final[:, 2],
#                                  mode='markers')])    
#np.savez("../output/q4_2.npz",F =F,M1 = M1,M2_final = M2,C1 = C1, C2_final = C2)
#plot(fig)
                   







