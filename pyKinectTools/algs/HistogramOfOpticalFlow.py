import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
import cv2

''' Taken with modifications from Scikit-Image version of HOG '''

''' Fix HOF: last orientation should be 'no motion' cell '''


def getFlow(imPrev, imNew):
    flow = cv2.calcOpticalFlowFarneback(imPrev, imNew, flow=None, pyr_scale=.5, levels=3, winsize=9, iterations=1, poly_n=3, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow

def getDepthFlow(imPrev, imNew):
    # Should actually go much more than 1 pixel!!!
    flow = np.zeros_like(imPrev)+999
    # flow = np.repeat(flow, 2, 2)
    
    # flow[im1==im2,:]=0
    flow[im1==im2]=4
    for x in xrange(1,im1.shape[0]):
        for y in xrange(1,im1.shape[1]):
            if flow[x,y]==999:
                flow[x,y] = np.argmin(im1[x-1:x+2, y-1:y+2]-im2[x-1:x+2, y-1:y+2])
    flow[flow==999] = -2

    flowNew = np.repeat(flow[:,:,np.newaxis], 2, 2)
    flowNew[flow==0,:] = [-1,-1]
    flowNew[flow==1,:] = [-1, 0]
    flowNew[flow==2,:] = [-1, 1]
    flowNew[flow==3,:] = [ 0,-1]
    flowNew[flow==4,:] = [ 0,0]
    flowNew[flow==5,:] = [ 0, 1]
    flowNew[flow==6,:] = [ 1,-1]
    flowNew[flow==7,:] = [ 1, 0]
    flowNew[flow==8,:] = [ 1, 1]
    return flow

def hog2image(hog, imageSize=[96,72],orientations=9,pixels_per_cell=(8, 8),cells_per_block=(3, 3)):
    from scipy import sqrt, pi, arctan2, cos, sin
    from skimage import draw

    sy, sx = imageSize
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1    

    hog = hog.reshape([n_blocksy, n_blocksx, by, bx, orientations])

    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = hog[y, x, :]
                orientation_histogram[y:y + by, x:x + bx, :] = block

    radius = min(cx, cy) // 2 - 1
    hog_image = np.zeros((sy, sx), dtype=float)
    for x in range(n_cellsx):
        for y in range(n_cellsy):
            for o in range(orientations):
                centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                dx = radius * cos(float(o) / orientations * np.pi)
                dy = radius * sin(float(o) / orientations * np.pi)
                # rr, cc = draw.bresenham(centre[0] - dy, centre[1] - dx,
                #                         centre[0] + dy, centre[1] + dx)
                rr, cc = draw.bresenham(centre[0] - dx, centre[1] - dy,
                                        centre[0] + dx, centre[1] + dy)  
                hog_image[rr, cc] += orientation_histogram[y, x, o]

    return hog_image


def showSplit(splitIm, blocks=[4,3]):
    for x in range(blocks[0]):
        for y in range(blocks[1]):
            i=y*4+x;
            subplot(4,3,i+1)
            imshow(splitIm[:,:,i])


# def splitIm(im, blocks=[4,3]):
#     subSizeX, subSizeY = im.shape / np.array(blocks)
#     newIms = np.empty([im.shape[0]/blocks[0], im.shape[1]/blocks[1], blocks[0]*blocks[1]])
#     for x in xrange(blocks[0]):
#         for y in xrange(blocks[1]):
#             newIms[:,:, x*blocks[1]+y] = im[x*subSizeX:(x+1)*subSizeX,y*subSizeY:(y+1)*subSizeY]

#     return newIms

def splitIm(im, blocks=[4,3]):
    subSizeX, subSizeY = im.shape / np.array(blocks)
    newIms = []
    for x in xrange(blocks[0]):
        for y in xrange(blocks[1]):
            newIms.append(im[x*subSizeX:(x+1)*subSizeX,y*subSizeY:(y+1)*subSizeY, :])

    newIms = np.dstack(newIms)
    return newIms    

from skimage import feature
def splitHog(im, blocks=[4,3], visualise=False):
    ims = splitIm(im, blocks)

    hogs = []
    hogIms = []
    for i in range(ims.shape[2]):
        if visualise:
            hogArray, hogIm = feature.hog(colorIm_g, visualise=True)
            hogs.append(hogArray)
            hogIms.append(hogArray)
        else:
            hogArray = feature.hog(colorIm_g, visualise=False)
            hogs.append(hogArray)
    
    if visualise:
        return hogs, hogIms
    else:
        return hogs


def hof(flow, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), visualise=False, normalise=False, motion_threshold=1.):

    """Extract Histogram of Optical Flow (HOF) for a given image.

    Key difference between this and HOG is that flow is MxNx2 instead of MxN


    Compute a Histogram of Optical Flow (HOF) by

        1. (optional) global image normalisation
        2. computing the dense optical flow
        3. computing flow histograms
        4. normalising across blocks
        5. flattening into a feature vector

    Parameters
    ----------
    Flow : (M, N) ndarray
        Input image (x and y flow images).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    visualise : bool, optional
        Also return an image of the hof.
    normalise : bool, optional
        Apply power law compression to normalise the image before
        processing.
    static_threshold : threshold for no motion

    Returns
    -------
    newarr : ndarray
        hof for the image as a 1D (flattened) array.
    hof_image : ndarray (if visualise=True)
        A visualisation of the hof image.

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA

    """
    flow = np.atleast_2d(flow)

    """ 
    -1-
    The first stage applies an optional global image normalisation
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    if flow.ndim < 3:
        raise ValueError("Requires dense flow in both directions")

    if normalise:
        flow = sqrt(flow)

    """ 
    -2-
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    if flow.dtype.kind == 'u':
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers in np.diff()
        flow = flow.astype('float')

    gx = np.zeros(flow.shape[:2])
    gy = np.zeros(flow.shape[:2])
    # gx[:, :-1] = np.diff(flow[:,:,1], n=1, axis=1)
    # gy[:-1, :] = np.diff(flow[:,:,0], n=1, axis=0)

    gx = flow[:,:,1]
    gy = flow[:,:,0]



    """ 
    -3-
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    magnitude = sqrt(gx**2 + gy**2)
    orientation = arctan2(gy, gx) * (180 / pi) % 180

    sy, sx = flow.shape[:2]
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    subsample = np.index_exp[cy / 2:cy * n_cellsy:cy, cx / 2:cx * n_cellsx:cx]
    for i in range(orientations-1):
        #create new integral image for this orientation
        # isolate orientations in this range

        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, -1)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                            temp_ori, -1)
        # select magnitudes for those orientations
        cond2 = (temp_ori > -1) * (magnitude > motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)

        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[:, :, i] = temp_filt[subsample]

    ''' Calculate the no-motion bin '''
    temp_mag = np.where(magnitude <= motion_threshold, magnitude, 0)

    temp_filt = uniform_filter(temp_mag, size=(cy, cx))
    orientation_histogram[:, :, -1] = temp_filt[subsample]

    # now for each cell, compute the histogram
    hof_image = None

    if visualise:
        from skimage import draw

        radius = min(cx, cy) // 2 - 1
        hof_image = np.zeros((sy, sx), dtype=float)
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o in range(orientations-1):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    dx = radius * cos(float(o) / orientations * np.pi)
                    dy = radius * sin(float(o) / orientations * np.pi)
                    rr, cc = draw.bresenham(centre[0] - dy, centre[1] - dx,
                                            centre[0] + dy, centre[1] + dx)
                    hof_image[rr, cc] += orientation_histogram[y, x, o]

    """
    The fourth stage computes normalisation, which takes local groups of
    cells and contrast normalises their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalise each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalised block descriptors as Histogram of Oriented
    Gradient (hog) descriptors.
    """

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                  by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y+by, x:x+bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / sqrt(block.sum()**2 + eps)

    """
    The final step collects the hof descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    if visualise:
        return normalised_blocks.ravel(), hof_image
    else:
        return normalised_blocks.ravel()
