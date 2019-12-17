import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input:
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car: [x1, y1, x2, y2]^T
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]

	# Put your implementation here

    ''' T correction didnt work because i didnt use the copy of P '''

    p = np.copy(p0)    # needed while calling LK for Template correction! 
    threshold = 0.01
    #random initialisation
    dp = 5
    
    h,w = It.shape

    It1_x = np.linspace(0, h-1, h)
    It1_y = np.linspace(0, w-1, w)
    spline_It1 = RectBivariateSpline(It1_x, It1_y, It1)


    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    
    #creation of Coordinates of Rectangle for warping I and T 
    YmeshRect, XmeshRect = np.meshgrid(np.arange(x1,x2), np.arange(y1,y2))
    XmeshRect = XmeshRect.flatten()
    YmeshRect = YmeshRect.flatten()
    XmeshRect_orig = np.copy(XmeshRect)
    YmeshRect_orig = np.copy(YmeshRect)
    

# A = w*h of rect,2
    

    A = np.zeros(((int(y2 - y1 + 1)*int(x2 - x1 + 1)), 2))
    b = np.zeros(((int(y2 - y1 + 1)*int(x2 - x1 + 1))))
    
    It_x = np.linspace(0, h-1, h)
    It_y = np.linspace(0, w-1, w)
    
    #template for using in b matrix  
    T = RectBivariateSpline(It_x, It_y, It).ev(XmeshRect_orig, YmeshRect_orig).flatten()
    
    while np.linalg.norm(dp) >= threshold:
    
    #warping updation deltaP 
        XmeshRect = XmeshRect_orig + p[0]
        YmeshRect = YmeshRect_orig + p[1]

    #warpedRect of Current Image
        Iwarped = spline_It1.ev(XmeshRect, YmeshRect)
        Iwarped = Iwarped.flatten()
    #gradients of Current Image for using in A matrix
        IwarpedDx = spline_It1.ev(XmeshRect, YmeshRect, dx=1)
        IwarpedDx = IwarpedDx.flatten()
        IwarpedDy = spline_It1.ev(XmeshRect, YmeshRect, dy=1)
        IwarpedDy = IwarpedDy.flatten()


        b = (T - Iwarped)
        A = np.asarray([IwarpedDx,IwarpedDy]).T
    #least square to find DeltaP 

        dp, a,b,c = np.linalg.lstsq(A, b)
        p += dp
		#print('p: ', p)

    return p
