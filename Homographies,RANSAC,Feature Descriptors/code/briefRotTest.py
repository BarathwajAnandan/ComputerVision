# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:00:26 2019

@author: KevinX
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

from BRIEF import briefLite,briefMatch,plotMatches




im1 = cv2.imread('../data/model_chickenbroth.jpg')
h,w = im1.shape[0:2]
locs1, desc1 = briefLite(im1)
locs2, desc2 = briefLite(im1)
matches = briefMatch(desc1, desc2)
plotMatches(im1,im1,matches,locs1,locs2)
match = []
angles =[]
for i in range(19):
    
    rotMat = cv2.getRotationMatrix2D((h/2,w/2), i*10 ,1.0)

    rotimg = cv2.warpAffine(im1,rotMat,(130,175))
    
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(rotimg)
    matches = briefMatch(desc1, desc2)
    #print(matches.shape)
    angle = i*10
    angles.append(angle)
    match.append(len(matches))

    
    
y_pos = np.linspace(0, 180, 19).astype(int)
#objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
# =============================================================================
# plt.bar(y_pos, match,width = 6.0)
# plt.xticks(y_pos, angles)
# 
# plt.xlabel('Rotation angle (degree)')
# plt.ylabel('Number of matches')
# plt.title('BRIEF Rotation Test')
# 
# =============================================================================
#plt.show()



    
    




