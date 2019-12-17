





import helper 
import numpy as np

import submission 

import matplotlib.pyplot as plt




img1 = plt.imread("../data/im1.png")
img2 = plt.imread("../data/im2.png")

h,w,c = img1.shape
points = np.load("../data/some_corresp.npz")
p1 = points['pts1'] 
p2 = points['pts2'] 
M = max (h,w)




F = submission.sevenpoint(p1,p2,M)

print(F.shape)




helper.displayEpipolarF(img1,img2,F)
