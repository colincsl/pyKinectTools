import numpy as np

''' 
---XYZ Convention---
x increases to the right
y increases upwards
z decreases as things get farther away
'''

''' Depth camera parameters '''
fx_d = 1.0 / 5.9421434211923247e+02;
fy_d = 1.0 / 5.9104053696870778e+02;
cx_d = 3.3930780975300314e+02;
cy_d = 2.4273913761751615e+02;

def depth2world(x_):
    ''' Convert depth coordinates to world coordinates standard Kinect calibration '''
    assert type(x_) == np.ndarray, "Wrong type into depth2world"

    if x_.shape[0]==76800 or (x_[:,0].max() < 240 and x_[:,1].max() < 320):
        x_[:,0] *= 2
        x_[:,1] *= 2
    
    y = np.array(x_[:,0])
    x = np.array(x_[:,1])
    d = np.array(x_[:,2])
    
    if np.all(d==0):
        return [0,0,0]

    xo =  ((x - cx_d) * d * fx_d)
    yo = -((y - cy_d) * d * fy_d)
    zo = -d
    return np.array([xo, yo, zo]).T    

def world2depth(x_, rez=[480, 640]):
    ''' 
    Convert world coordinates to depth coordinates using the standard Kinect calibration 
    ---Parameters---
    x_ : numpy array (Nx3)
    rez : resolution of depth image
    '''
    assert type(x_) == np.ndarray, "world2depth input must be a numpy array"

    y = np.array(x_[:,1])     
    x = np.array(x_[:,0])
    ''' Make sure z coordinates are positive and greater than 0'''
    z = np.abs(np.array(x_[:,2]))
    z = np.maximum(1, z)

    xo = (( x / (z * fx_d)) + cx_d).astype(np.int)
    yo = ((-y / (z * fy_d)) + cy_d).astype(np.int)

    if rez != [480,640]:
        yo *= rez[0] / 480.
        xo *= rez[1] / 640.        

    return np.array([xo, yo, z], dtype=np.int).T


def depthIm2XYZ(depthMap):
    inds = np.nonzero(depthMap>0)
    xyzVals = depth2world(np.array([inds[0],inds[1],depthMap[inds[0],inds[1]]]).T)

    return xyzVals

def depthIm2PosIm(depthMap):
    imOut = np.zeros([depthMap.shape[0],depthMap.shape[1],3], dtype=float)
    imOut[:,:,2] = depthMap

    inds = np.nonzero(imOut[:,:,2]>0)
    xyzVals = depth2world(np.array([inds[0],inds[1],imOut[inds[0],inds[1],2]]).T)

    imOut[inds[0],inds[1],0] = xyzVals[:,0]
    imOut[inds[0],inds[1],1] = xyzVals[:,1]

    return imOut


def posImage2XYZ(im_in, min_=500, max_=4000):
    # Used to convert 8bit depth image to a
    # 3 dimensional floating point image w/ x,y,z
    imOut = np.zeros([im_in.shape[0],im_in.shape[1],3], dtype=float)
    if im_in.sum() == 0:
        return imOut

    imOut[:,:,2] = im_in
    imOut[:,:,2] *= float((max_-min_)/256.0)
    imOut[:,:,2] += min_
    imOut[:,:,2] *= imOut[:,:,2]>min_    

    inds = np.nonzero(imOut[:,:,2]>min_)
    xyzVals = depth2world(np.array([inds[0],inds[1],imOut[inds[0],inds[1],2]]).T)

    # imOut[inds] = xyzVals
    imOut[inds[0],inds[1],0] = xyzVals[:,0]
    imOut[inds[0],inds[1],1] = xyzVals[:,1]
    # imOut[inds[0],inds[1],0] = xyzVals[:,1]
    # imOut[inds[0],inds[1],1] = xyzVals[:,0]    
    imOut[inds[0],inds[1],2] = xyzVals[:,2]

    return imOut


