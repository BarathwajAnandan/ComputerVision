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
rect = np.array([59,116, 145, 151])
rect0 = np.array([59,116, 145, 151])
rectList = np.empty((0, 4))




for i in range(0, (f-1)):
    print("vid: ", i)
    mod_p = 0
    #prev frame = template , currrent frame = current image
    rectList = np.append(rectList, rect.reshape((1,4)), axis=0)
    p = LucasKanade(vid[:,:,i], vid[:,:,i+1], rect)
    offset =  rect - rect0
    mod_p += p + np.flip(offset[0:2])
    
    p0 = np.asarray([rect[1] - rect0[1]+ p[0], rect[0] + p[1] - rect0[0]]).reshape(2)
    
    #print(p0,mod_p)
    pstar = LucasKanade(vid[:,:,0], vid[:,:,i+1],rect0,p0) -p0
    
    
    
    
    if 0.1>np.linalg.norm(pstar-p0) :
        rect[0] += p[1]
        rect[1] += p[0]
        rect[2] +=  p[1]
        rect[3] +=  p[0]
    else:
        rect[0] += pstar[1]
        rect[1] += pstar[0]
        rect[2] +=  pstar[1]
        rect[3] +=  pstar[0]
        
        
np.save('carseqrects-wcrt.npy', rectList)

# Playback vids with rectangles


rectList = np.load('carseqrects-wcrt.npy')


    
for i in range(vid.shape[2]-1)  :  
    img = vid[:,:,i].copy()
    rect = rectList[i, :]
    
    plt.gca().add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
    #ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
    plt.imshow(img, cmap='gray')
    plt.show()
    
