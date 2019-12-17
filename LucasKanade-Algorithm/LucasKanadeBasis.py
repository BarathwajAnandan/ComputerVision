# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 02:23:40 2019

@author: KevinX
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect,bases, p0 = np.zeros((2))):
	# Input:
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car: [x1, y1, x2, y2]^T
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
    template  = It
    h,w = template.shape
    h1,w1 = It1.shape
	# Put your implementation here
    hb,wb,countb = bases.shape
    
    bases = np.reshape(bases,(hb*wb,countb))
    
    BBT = bases @ bases.T
    
    threshold = 0.001
    
    It1_x = np.linspace(0, h1-1, h1)
    It1_y = np.linspace(0, w1-1, w1)
    spline_It1 = RectBivariateSpline(It1_x, It1_y, It1)

    template_x = np.linspace(0, h-1, h)
    template_y = np.linspace(0, w-1, w)
    spline_template = RectBivariateSpline(template_x, template_y, template)
    B_ortho = np.identity(len(BBT)) - BBT 


# x,yT , x1,y1T  (Thus the element swap)
    y1 = rect[0]
    x1 = rect[1]
    y2 = rect[2]
    x2 = rect[3]

    width = (np.rint(x2 - x1 + 1))
    height =(np.rint(y2 - y1 + 1))
    A_row = int(width*height)
    print(height,width)
    
    #print(A_row)

    YmeshRect, XmeshRect = np.meshgrid(np.arange(height), np.arange(width))
    YmeshRect+= y1
    XmeshRect+= x1
    
    
    #YmeshRect.shape

    XmeshRect = np.squeeze(XmeshRect)
    YmeshRect = np.squeeze(YmeshRect)
    XmeshRect_orig = np.reshape(XmeshRect, (1, XmeshRect.shape[0]*XmeshRect.shape[1]))
    YmeshRect_orig = np.reshape(YmeshRect, (1, YmeshRect.shape[0]*YmeshRect.shape[1]))
    print(XmeshRect.shape,XmeshRect_orig.shape)

    #A = np.zeros((A_row, 2))
    
    #print(A.shape)
    
    
    b = np.zeros((A_row))

    I_orig = spline_template.ev(XmeshRect_orig, YmeshRect_orig)
    I_orig = I_orig.squeeze()
    
    #print(I_orig.shape)

    dp_norm = threshold
    while dp_norm >= threshold:

        XmeshRect = XmeshRect_orig + p0[0]
        XmeshRect_orig = XmeshRect_orig.squeeze()
        YmeshRect = YmeshRect_orig + p0[1]
        YmeshRect_orig = YmeshRect_orig.squeeze()

        I = spline_It1.ev(XmeshRect, YmeshRect)
        I = I.squeeze()
        Ix = spline_It1.ev(XmeshRect, YmeshRect, dx=1)
        Ix = Ix.squeeze()
        Iy = spline_It1.ev(XmeshRect, YmeshRect, dy=1)
        Iy = Iy.squeeze()
		# Construct A and b matrices
        b = (I_orig - I)
        A = np.asarray([Ix,Iy]).T
        
       # print (A *A.T)
        
#        print(A.shape)
#        print(b.shape)
#        print(bases.shape)    

        #print(B_ortho.shape ,b.shape ,A.shape)          
        Bb = B_ortho@b
        BA = B_ortho@A
         

        delP, e, rank, s = np.linalg.lstsq(BA, Bb)

        dp_norm = np.linalg.norm(delP)
		#print('dp_norm: ', dp_norm)
        p0 += delP
		#print('p: ', p)

    return p0
