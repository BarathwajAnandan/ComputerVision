import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches



# write your script here, we recommend the above libraries for making your animation

from SubtractDominantMotion import SubtractDominantMotion

vid = np.load("../data/aerialseq.npy")
AerialSeq = (np.zeros((vid.shape[0],vid.shape[1])))
mask = np.ones((vid.shape[0],vid.shape[1]))

save_mask = []
h,w,f = vid.shape
for i in range(f-1):
    print(i)
    current_frame = vid[:,:,i]
    next_frame = vid[:,:,i+1]
    
    frame_rgb = np.stack((next_frame,next_frame,next_frame),axis = 2)
    #print(mask.shape)
    fig = plt.imshow(frame_rgb,cmap = 'gray')
    plt.title("frame:" + str(i))
    mask = SubtractDominantMotion(current_frame,next_frame)
    mask_rgb = np.stack((mask*0,mask*255,mask*255),axis = 2) 
    fig =plt.imshow(mask_rgb,alpha=0.5)   
    #print(i)
# =============================================================================
#     if i in [30,60,90,120]:
#         save_mask.append(mask)
#    plt.show()
# =============================================================================
#np.save("aerialseqrects.npy",save_mask)
#print(save_mask.shape)
    plt.show()

    