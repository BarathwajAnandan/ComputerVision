import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
# write your script here, we recommend the above libraries for making your animation

from LucasKanade import LucasKanade

vid = np.load('../data/carseq.npy')
#vid = np.load('../data/carseq.npy')
h,w,f = vid.shape
#rect = np.array([101, 61, 155, 107]).astype(float).T
rect = np.array([59,116, 145, 151]).astype(float).T
rectList = np.empty((0, 4))




for i in range(0, (f-1)):
    print("vid: ", i)
    #prev frame = template , currrent frame = current image
    rectList = np.append(rectList, rect.reshape((1,4)), axis=0)
    p = LucasKanade(vid[:,:,i], vid[:,:,i+1], rect)
    
    #print(p, np.flip(p))
    rect[0] += p[1]
    rect[1] += p[0]
    rect[2] +=  p[1]
    rect[3] +=  p[0]
np.save('carseqrects.npy', rectList)

# Playback vids with rectangles
rectList = np.load('carseqrects.npy')


#for i in [1,100,200,300,400]:
    
for i in range(vid.shape[2]-1)  :  
    img = vid[:,:,i].copy()
    rect = rectList[i, :]
    
# =============================================================================
#     plt.gca().add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
#     #ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
#     plt.imshow(img, cmap='gray')
#     plt.show()
#     
#     plt.pause(0.01)
# #plt.pause(1)
#     
# #plt.show(img)
#     
# 
# =============================================================================
