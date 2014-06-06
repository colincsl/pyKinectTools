
''' 
*Calculate normals from XYZ data
*RANSAC-based Surface detection

Colin Lea
pyKinectTools
'''

import numpy as np
import cv2
import skimage.morphology
import numpy as np
from pyKinectTools.utils.DepthUtils import *

def compute_normals(im_pos, n_offset=3):
    """
    Converts an XYZ image to a Normal Image
    --Input--
    im_pos : ndarray (NxMx3)
        Image with x/y/z values for each pixel
    n_offset : int
        Smoothness factor for calculating the gradient
    --Output--
    normals : ndarray (NxMx3)
        Image with normal vectors for each pixel
    """
    gradients_x = np.zeros_like(im_pos)
    gradients_y = np.zeros_like(im_pos)
    for i in range(3):
        gradients_x[:,:,i], gradients_y[:,:,i] = np.gradient(im_pos[:,:,i], n_offset)

    gradients_x /= np.sqrt(np.sum(gradients_x**2, -1))[:,:,None]
    gradients_y /= np.sqrt(np.sum(gradients_y**2, -1))[:,:,None]

    normals = np.cross(gradients_x.reshape([-1, 3]), gradients_y.reshape([-1, 3])).reshape(im_pos.shape)
    normals /= np.sqrt(np.sum(normals**2, -1))[:,:,None]
    normals = np.nan_to_num(normals)
    return normals  


def compute_normals_with_radius(posMat, radius=3):
    """
    (deprecated: This function is much slower than compute_normals())

    Converts a set of XYZ points a set of normals
    --Input--
    im_pos : ndarray (NxMx3)
        Image with x/y/z values for each pixel
    n_offset : int
        Smoothness factor for calculating the gradient
    --Output--
    normals : ndarray (NxMx3)
        Image with normal vectors for each pixel
    """
    print "compute_normals_with_radius() is deprecated: This function is much slower than compute_normals()"

	assert len(posMat.shape)== 3, "Input should be an NxMx3 posMat"
	
	height = posMat.shape[0]
	width = posMat.shape[1]
	normals = np.zeros_like(posMat)

	for y in range(radius, height-radius):
		for x in range(radius, width-radius):
			patch = posMat[y-radius:y+radius+1, x-radius:x+radius+1, :].reshape([-1,3])

			if np.sum(np.sum(patch == 0, 1) != 3) > 3:
				mean = patch.mean(0)
				_,_,vT = np.linalg.svd(patch-mean, full_matrices=False)
				normals[y,x,:] = vT[2,:]

	return normals


def getTopdownMap(depthMap, rotation=None, rez=1000, centroid=[0,0,0], bounds=[]):
	"""
	Transform a pointcloud with arbitrary rotation and show output it's top-down view as an image
	"""
	# ie. bounds=[4000,4000,2000]	

	xyz = depthIm2XYZ(depthMap)

	if centroid == []:
		centroid = xyz.mean(0)
	xyz -= centroid

	if rotation is not None:
		xyzNew = np.asarray(rotation*np.asmatrix(xyz.T)).T
	else:
		xyzNew = xyz
	xyzNew += centroid
	xyzMin = xyzNew.min(0)
	xyzMax = xyzNew.max(0)

	if bounds == []:
		bounds = xyzMax - xyzMin

	# Top-down view
	indsNew = np.asarray([np.round((xyzNew[:,0] - xyzMin[0])/(xyzMax[0]-xyzMin[0])*(rez-1)),
				np.round((xyzNew[:,1] - xyzMin[1])/(xyzMax[1]-xyzMin[1])*(rez-1))], dtype=np.int)
	# indsNew = np.asarray([np.round((xyzNew[:,0] + bounds[0]/2)/bounds[0]*(rez-1)),
	# 			np.round((xyzNew[:,1] + bounds[1]/2)/(bounds[1])*(rez-1))], dtype=np.int)

	indsNew[indsNew < 0] = 0
	indsNew[indsNew >= rez] = 0
	posMatNew = np.zeros([rez, rez, 3])
	posMatNew[indsNew[0], indsNew[1]] = (xyzNew-xyzMin)/(xyzMax-xyzMin)

	return posMatNew

def getSceneOrientation(posMat, coordsStart=[350,350], coordsEnd=[425,425]):
	"""
	Calculate the surface from a patch in a XYZ image
	"""
	inds = np.nonzero(posMat[coordsStart[0]:coordsEnd[0], coordsStart[1]:coordsEnd[1], 2])
	inds = [inds[0] + coordsStart[0], inds[1] + coordsStart[0]]
	xyzFloor = posMat[inds[1], inds[0]]
	meanFloor = xyzFloor.mean(0)
	xyzFloor -= meanFloor
	U, _, vT = np.linalg.svd(xyzFloor, full_matrices=False)

	return vT


def plane_detection(im_pos, im_norm, thresh=30, n_iter=None):
    """ RANSAC-based plane detection 

    """
    n_points = im_norm.shape[0]*im_norm.shape[1]
    pts = im_pos.reshape([-1, 3])
    pts_normals = im_norm.reshape([-1, 3])
    valid_pts = np.all(pts_normals!=0, -1)
    valid_pts_idx = np.nonzero(valid_pts)[0]
    n_valid_pts = valid_pts_idx.shape[0]

    if n_iter is None:
        n_iter = int(np.sqrt(n_valid_pts))
        # pct_inliers = .2
        # n_req_points = 3
        # n_iter = int(np.log(1-.99) / np.log(1 - pct_inliers**n_req_points))
    
    best_n_inliers = 0
    best_offset = None
    best_normal = None
    best_valid = None
    # Get random initial points
    random_pts_idx = np.random.randint(0, n_valid_pts, n_iter)
    # n_iter = 1
    # i=0
    for i in range(n_iter):
        # Get random point/normal
        pt_idx = random_pts_idx[i]
        pt = pts[valid_pts_idx[pt_idx]]
        normal = pts_normals[valid_pts_idx[pt_idx]]

        # Find distance along normal direction from point
        dist_to_plane = np.abs(np.dot((pts - pt), normal))
        # Check if the points are close enough to the plane's surface. And remove NaNs
        valid = (dist_to_plane < thresh) * valid_pts
        n_inliers = np.sum(valid) 

        if n_inliers > best_n_inliers:
            # Compute better offset and normal
            best_offset = np.mean(pts[valid], 0)
            U,S,Vt = np.linalg.svd(pts[valid] - best_offset, full_matrices=False)
            best_normal = Vt[2]
            
            # Compute distance to new plane and find valid points
            dist_to_plane = np.abs(np.dot((pts - best_offset), best_normal))
            valid = dist_to_plane < thresh
            
            best_n_inliers = np.sum(valid*valid_pts) 
            best_valid = valid

    return best_offset, best_normal, best_valid


def plane_detection_pts(pts, pts_normals, thresh=30, n_iter=None):
    """ RANSAC-based plane detection """
    n_pts = pts.shape[0]
    valid_pts = np.all(pts_normals!=0, -1)

    if n_iter is None:
        n_iter = int(np.sqrt(n_pts))

    best_n_inliers = 0
    best_offset = None
    best_normal = None
    best_valid = None

    # Get random initial points
    random_pts_idx = np.random.randint(0, n_pts, n_iter)

    for i in range(n_iter):
        # Get random point/normal
        pt_idx = random_pts_idx[i]
        pt = pts[pt_idx]
        normal = pts_normals[pt_idx]

        # Find distance along normal direction from point
        dist_to_plane = np.abs(np.dot((pts - pt), normal))

        # Check if the points are close enough to the plane's surface. And remove NaNs
        valid = dist_to_plane < thresh
        n_inliers = np.sum(valid) 

        if n_inliers > best_n_inliers:
            # Compute better offset and normal
            best_offset = np.mean(pts[valid], 0)
            U,S,Vt = np.linalg.svd(pts[valid] - best_offset, full_matrices=False)
            best_normal = Vt[2]
            
            # Compute distance to new plane and find valid points
            dist_to_plane = np.abs(np.dot((pts - best_offset), best_normal))
            valid = dist_to_plane < thresh
            
            best_n_inliers = np.sum(valid) 
            best_valid = valid

    return best_offset, best_normal, np.sum(best_valid)


def compute_foreground_from_surface(im_pos, offset, normal, thresh=30):
	"""
	Given a plane, check which points in an XYZ image are valid
	"""
    im_shape = im_pos.shape[:2]
    dist_to_plane = np.abs(np.dot((im_pos.reshape([-1,3]) - offset), normal))
    valid = dist_to_plane < thresh
    valid = valid.reshape(im_shape)

    # remove error points
    # valid *= (im_pos[:,:,2]!=0) 

    return valid

def tabletop_detector(im_depth, bg_threshold=1000, subsample=1, im_pos=None, im_norm=None, thresh=30, n_iter=100):
    """
    --Input--
    im_depth : ndimage
    bg_threshold : int
        cuttoff the back of the image
    subsample : int
        Downsample image? 

    --Output--
    plane_offset : ndarray
    	centroid of the plane
    plane_normal : ndarray
    	surface normal of the plane
    """

    # Remove points that are far away
    if bg_threshold is not None:
        mask_depth = (im_depth < bg_threshold) * (im_depth != 0)
    else:
        mask_depth = im_depth != 0
    
    # Resize depth (for speed)
    if subsample > 1:
	    im_depth = im_depth[::subsample,::subsample]
    	mask_depth = mask_depth[::subsample,::subsample]

    # Get surface normals
    if im_pos is None:
        im_pos = depthIm2PosIm(im_depth)
    if im_norm is None:
        im_norm = compute_normals(im_pos, n_offset=10)
    im_pos[:,:,2][~mask_depth] = 0
    
    im_shape = im_depth.shape
    pos_shape = im_pos.shape

    # Get foreground from ground plane detection
    plane_offset, plane_normal, _ = plane_detection(im_pos[::subsample,::subsample], im_norm[::subsample,::subsample], thresh=thresh, n_iter=n_iter)

    return plane_offset, plane_normal

