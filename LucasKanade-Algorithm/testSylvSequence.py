import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
# write your script here, we recommend the above libraries for making your animation

from LucasKanadeBasis import LucasKanadeBasis

vid = np.load('../data/sylvseq.npy')
h,w,f = vid.shape
#101; 61; 155; 107
rect = np.array([101, 61, 155, 107]).T.astype(float)


rectList = np.empty((0, 4))
bases = np.load("../data/sylvbases.npy")

print(bases.shape)



for i in range(0, f-1):
    print("vid: ", i)
    #prev frame = template , currrent frame = current image
    rectList = np.append(rectList, rect.reshape((1,4)), axis=0)
    p = LucasKanadeBasis(vid[:,:,i], vid[:,:,i+1], rect, bases)
    rect[0] += p[1]
    rect[1] += p[0]
    rect[2] +=  p[1]
    rect[3] +=  p[0]

    
    
# =============================================================================
#     ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
#     ax.imshow(vid[:,:,i+1], cmap='gray')
#     #plt.pause(0.01)
#     ax.clear()
# =============================================================================
#np.save('sylvseqrects.npy', rectList)

# Playback vids with rectangles
rectList = np.load('sylvseqrects.npy')
#rectList1 = np.load('sylvseqLK.npy')


for i in range(vid.shape[2]-1)  :  
    img = vid[:,:,i].copy()
    rect = rectList[i, :]
    #rect1 = rectList1[i, :]
    
    plt.gca().add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
    #plt.gca().add_patch(patches.Rectangle((rect1[0], rect1[1]), rect1[2]-rect1[0]+1, rect1[3]-rect1[1]+1, linewidth=2, edgecolor='green', fill=False))
    
    
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.pause(0.01)
    
#    plt.imsave("sylvOutput" + str(i) + ".jpg",img)
#plt.show(img)
    
