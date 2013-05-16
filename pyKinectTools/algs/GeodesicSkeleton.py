import numpy as np
import scipy.ndimage as nd
import pyKinectTools.algs.Dijkstras as dgn

# from pyKinectTools.utils.DepthUtils import *
from pyKinectTools.utils.DepthUtils import depthIm2PosIm
from copy import deepcopy
from skimage.draw import circle

from IPython import embed
from pylab import *

def geodesic_extrema_MPI(im_pos, centroid=None, iterations=1, visualize=False, box=None):
	'''
	im : im_pos (NxMx3)
	'''
	if centroid==None:
		centroid = np.array(nd.center_of_mass(im_pos[:,:,2]), dtype=np.int16)
	if box is not None:
		im_pos = im_pos[box]
	im_pos = np.ascontiguousarray(im_pos, dtype=np.int16)

	if visualize:
		cost_map = np.zeros([im_pos.shape[0], im_pos.shape[1]], dtype=np.uint16)
		extrema = dgn.geodesic_map_MPI(cost_map, im_pos, np.array(centroid, dtype=np.int16), iterations, 1)
		cost_map = np.array(extrema[-1])
		extrema = extrema[:-1]
		extrema = np.array([x for x in extrema])
		return extrema, cost_map
	else:
		extrema = np.array(dgn.geodesic_extrema_MPI(im_pos, np.array(centroid, dtype=np.int16), iterations))
		return extrema

def connect_extrema(im_pos, target, markers, visualize=False):
	'''
	im_pos : XYZ positions of each point in image formation (n x m x 3)
	'''
	height, width,_ = im_pos.shape
	centroid = np.array(target)

	im_pos = np.ascontiguousarray(im_pos.astype(np.int16))
	cost_map = np.ascontiguousarray(np.zeros([height, width], dtype=np.uint16))

	extrema = dgn.geodesic_map_MPI(cost_map, im_pos, np.array(centroid, dtype=np.int16), 1, 1)
	cost_map = extrema[-1]

	trails = []
	for m in markers:
		trail = dgn.geodesic_trail(cost_map.copy()+(32000*(im_pos[:,:,2]==0)).astype(np.uint16), np.array(m, dtype=np.int16))
		trails += [trail.copy()]
	if visualize:
		cost_map = deepcopy(cost_map)
		circ = circle(markers[0][0],markers[0][1], 5)
		circ = np.array([np.minimum(circ[0], height-1), np.minimum(circ[1], width-1)])
		circ = np.array([np.maximum(circ[0], 0), np.maximum(circ[1], 0)])
		cost_map[circ[0], circ[1]] = 0
		for i,t in enumerate(trails[1:]):
			# embed()
			cost_map[t[:,0], t[:,1]] = 0
			circ = circle(markers[i+1][0],markers[i+1][1], 5)
			circ = np.array([np.minimum(circ[0], height-1), np.minimum(circ[1], width-1)])
			circ = np.array([np.maximum(circ[0], 0), np.maximum(circ[1], 0)])
			cost_map[circ[0], circ[1]] = 0
		return trails, cost_map
	else:
		return trails



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


''' --- Other functions --- '''

def relative_marker_positions(im_pos, markers):
	feature_map = np.zeros([im_pos.shape[0], im_pos.shape[1], len(markers)], dtype=np.float)
	dgn.relative_distance_features(np.ascontiguousarray(im_pos), np.ascontiguousarray(markers), feature_map)
	return feature_map

def local_histograms(im, n_bins=2, max_bound=255, patch_size=5):
	output = dgn.local_histograms(np.ascontiguousarray(im), n_bins, patch_size, max_bound)
	return output


