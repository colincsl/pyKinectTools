import numpy as np
import scipy.ndimage as nd
import pyKinectTools.algs.Dijkstras as dgn

from pyKinectTools.utils.DepthUtils import *
from pyKinectTools.utils.DepthUtils import depthIm2PosIm
from IPython import embed
from pylab import *

def geodesic_extrema_MPI(im, centroid=None, iterations=1):
	if centroid==None:
		centroid = np.array(nd.center_of_mass(im), dtype=np.int)
	posmat = depthIm2PosIm(im).astype(np.int16)
	# cost_map = dgn.geodesic_extrema_MPI(posmat, np.array(centroid, dtype=np.int16), iterations)
	# return cost_map
	extrema = dgn.geodesic_extrema_MPI(posmat, np.array(centroid, dtype=np.int16), iterations)
	extrema = np.vstack([centroid, extrema])
	# extrema = extrema[:,[1,0]]
	return extrema


def distance_map(im, centroid, scale=1):
	'''
	---Parameters---
	im_depth : 
	centroid : 
	---Returns---
	distance_map
	'''

	im_depth = np.ascontiguousarray(im.copy())
	objSize = im_depth.shape
	max_value = 32000
	mask = im_depth > 0

	# Get discrete form of position/depth matrix
	# embed()
	depth_min = im_depth[mask].min()
	depth_max = im_depth[mask].max()
	depth_diff = depth_max - depth_min
	if depth_diff < 1:
		depth_diff = 1
	scale_to = scale / float(depth_diff)

	# Ensure the centroid is within the boundaries
	# Segfaults if on the very edge(!) so set border as 1 to resolution-2
	centroid[0] = centroid[0] if centroid[0] > 0 else 1
	centroid[0] = centroid[0] if centroid[0] < im.shape[0]-1 else im.shape[0]-2
	centroid[1] = centroid[1] if centroid[1] > 0 else 1
	centroid[1] = centroid[1] if centroid[1] < im.shape[1]-1 else im.shape[1]-2	

	# Scale depth image
	im_depth_scaled = np.ascontiguousarray(np.array( (im_depth-depth_min)*scale_to, dtype=np.uint16))
	# im_depth_scaled = np.ascontiguousarray(np.array( (im_depth-depth_min), dtype=np.uint16))
	im_depth_scaled *= mask

	# Initialize all but starting point as max
	distance_map = np.zeros([objSize[0],objSize[1]], dtype=np.uint16)+max_value	
	distance_map[centroid[0], centroid[1]] = 0

	# Set which pixels are in/out of bounds
	visited_map = np.zeros_like(distance_map, dtype=np.uint8)
	visited_map[-mask] = 255

	centroid = np.array(centroid, dtype=np.int16)
	# embed()
	dgn.distance_map(distance_map, visited_map, im_depth_scaled.astype(np.uint16), centroid, int(scale))

	return distance_map.copy()



def generateKeypoints(im, centroid, iterations=10, scale=6):
	'''
	---Parameters---
	im_depth : 
	centroid : 
	---Returns---
	extrema
	distance_map
	'''

	# from pylab import *
	# embed()


	x,y = centroid
	maps = []
	extrema = []
	# Get N distance maps. For 2..N centroid is previous farthest distance.
	for i in range(iterations):
		im_dist = distance_map(np.ascontiguousarray(im.copy()), centroid=[x,y], scale=scale)
		im_dist[im_dist>=32000] = 0
		maps += [im_dist.copy()]
		max_ = np.argmax(np.min(np.dstack(maps),-1))
		max_px = np.unravel_index(max_, im.shape)
		x,y = max_px
		extrema += [[x,y]]

	im_min = np.min(np.dstack(maps),-1)
	im_min = (im_min/(float(im_min.max())/255.)).astype(np.uint8)
	# Visualize
	im -= im[im>0].min()
	for c in extrema:
		# im_min[c[0]-3:c[0]+4, c[1]-3:c[1]+4] = im_min.max()
		im[c[0]-3:c[0]+4, c[1]-3:c[1]+4] = im.max()


	import cv2
	cv2.imshow("Extrema", im/float(im.max()))
	# cv2.imshow("Extrema", im_min/float(im_min.max()))
	cv2.waitKey(30)

	return extrema, im_min


def drawTrail(im, trail, value=255):
	for i,j in trail:
		im[i,j] = value
	return im

