from pylab import *
import numpy as np
import cv

''' 
---XYZ Convention---
x increases to the left
y increases upwards
z increases as things get farther away
'''

# Depth camera parameters
fx_d = 1.0 / 5.9421434211923247e+02;
fy_d = 1.0 / 5.9104053696870778e+02;
cx_d = 3.3930780975300314e+02;
cy_d = 2.4273913761751615e+02;

def depth2world(x_, y=0, d=0):
    if type(x_) == list:
        x = np.array(x_[0])        
        y =-np.array(x_[1])
        d = np.array(x_[2])
    elif type(x_) == np.ndarray:
        x = 640 - np.array(x_[:,1])   
        y = 480 - np.array(x_[:,0])
        # y = 480 - np.array(x_[:,0])
        d = np.array(x_[:,2])
        # d = np.maximum(1, d)
    else:
        x = x_
    if np.all(d==0):
        return [0,0,0]
    xo = ((x - cx_d) * d * fx_d)
    yo = ((y - cy_d) * d * fy_d)
    zo = d
    return np.array([xo, yo, zo]).T

def world2depth(x_, y=0, z=0):
    if type(x_) == list:
        y = np.array(x_[1])#*10
        z = np.array(x_[2])
        x = np.array(x_[0])#*10  
        if z == 0:
            return [0,0,0]
        # z = np.maximum(1, z)
    elif type(x_) == np.ndarray:
        y = np.array(x_[:,1])#*10
        z = np.array(x_[:,2])
        x = np.array(x_[:,0])#*10
        z = np.maximum(1, z)
    else:
        x  = x_

    yo = 640 - np.array((np.round(x / fx_d / z + cx_d)), dtype=np.int)
    xo = 480 - np.array((np.round((y / fy_d / z) + cy_d)), dtype=np.int)
    do = np.array(z, dtype=np.int)
    return np.array([xo, yo, do])

def posImage2XYZ(im_in, min_=500, max_=4000):
    # Used to convert 8bit depth image to a 
    # 3 dimensional floating point image w/ x,y,z
    imOut = np.zeros([im_in.shape[0],im_in.shape[1],3], dtype=float)
    imOut[:,:,2] = im_in
    imOut[:,:,2] *= float((max_-min_)/256)
    imOut[imOut[:,:,2]>min_,2] += min_
    imOut[:,:,2] *= imOut[:,:,2]>min_

    inds = np.nonzero(imOut[:,:,2]>min_)
    xyzVals = depth2world(np.array([inds[0],inds[1],imOut[inds[0],inds[1],2]]).T)

    # imOut[inds] = xyzVals
    imOut[inds[0],inds[1],0] = xyzVals[:,1]
    imOut[inds[0],inds[1],1] = xyzVals[:,0]
    imOut[inds[0],inds[1],2] = xyzVals[:,2]

    return imOut


