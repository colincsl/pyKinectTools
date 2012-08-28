'''


pyKinectTools
Colin Lea
'''

# from pylab import *
import numpy as np
import scipy.ndimage as nd
##### pyvision library (w/ HOG algorithm)
import sys
sys.path.append("/Users/colin/libs/pyvision/build/lib.macosx-10.7-intel-2.7/")
from vision import features 
#####

''' Make picture of positive HOG weights.'''
def HOGpicture(w, bs=8, positive=True):
    # w=feature, bs=size, 
    # construct a "glyph" for each orientaion
    bim1 = np.zeros([bs,bs])
    bim1[:,round(bs/2):round(bs/2)+1] = 1

    bim = np.zeros([bim1.shape[0],bim1.shape[1], 9])
    bim[:,:,0] = bim1
    for i in range(2,10):
        bim[:,:,i-1] = nd.rotate(bim1, -(i-1)*20, reshape=False) #crop?

    # make pictures of positive weights bs adding up weighted glyphs
    shape_ = w.shape
    if positive:
        w[w<0] = 0
    else:
        w[w>0] = 0
    # im = np.zeros([bs*shape_[0], bs*shape_[1]])
    im = np.zeros([bs*shape_[1], bs*shape_[0]])
    for i in range(0,shape_[1]):
        for j in range(0,shape_[0]):
            for k in range(9):
                # pass
                # im[(i-1)*bs:i*bs, (j-1)*bs:j*bs] += bim[:,:,k]*w[j,i,k]
                im[(i)*bs:(i+1)*bs, (j)*bs:(j+1)*bs] += bim[:,:,k]*w[j,i,k]
                # im[(j-1)*bs:j*bs,(i-1)*bs:i*bs] += bim[:,:,k]*w[i,j,k]


    return im


def overlayHOG(im_, hogIm):
    im = np.copy(im_)*1.0
    imShape = im.shape
    hShape = hogIm.shape
    xMin = np.floor((imShape[0]-hShape[0]))
    xMax = imShape[0]
    yMin = np.floor((imShape[1]-hShape[1]))
    yMax = imShape[1]
    # print xMin, xMax
    # print yMin, yMax

    im[xMin:xMax, yMin:yMax] = np.maximum(im[xMin:xMax, yMin:yMax], hogIm*10.0)

    return im