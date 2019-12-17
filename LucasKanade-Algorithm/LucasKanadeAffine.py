import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform as af

def LucasKanadeAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrgradientX [2x3 numpy array]
	# put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.array([[0.0, 0.0, 0.0],[ 0.0, 0.0, 0.0]])

    h,w = It.shape#width = It.shape[1]

    x, y = np.meshgrid( np.linspace(0,h-1,h), np.linspace(0,w-1,w))
    x = x.flatten()
    y = y.flatten()#height = It.shape[0]
    gradientY, gradientX = np.gradient(It1)

    validity = np.ones((h, w))

    threshold = 2
    dp_norm = threshold

    while dp_norm >= threshold:

        It = It.flatten()
        warped_img =af(It1, M ).flatten()
        warpedGx = af(gradientX, M).flatten()
        warpedGy = af(gradientY, M).flatten()
        #warped_mask = cv2.warpAffine(mask, M, (It.shape[1], It.shape[0]))

     
        
        # [x y 1 0 0 0
        #   0 0 0 x y 1] multiplied by GradientI

        A = [warpedGx*x,warpedGx*y,warpedGx, warpedGy*x,warpedGy*y,warpedGy]
        A = np.asarray(A).squeeze().T
        b = (warped_img - It).reshape((w*h, 1))

        dp, a,b,c = np.linalg.lstsq(A, b)
        dp = np.reshape(dp,(2,3))
        dp_norm = np.linalg.norm(dp)
        #print('dp_norm: ', dp_norm)
        p += dp
        M = p
        
        M[0,0] += 1
        M[1,1] += 1
        i = np.asarray([0,0,1])
        M = np.append(M,i).reshape((3,3))
        print(M.shape)
        
        
            	#print(M)
        #print(M)      #print('p: ', p)
    return M
