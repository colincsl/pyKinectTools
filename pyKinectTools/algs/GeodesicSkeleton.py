import numpy as np
import scipy.ndimage as nd
import pyKinectTools.algs.Dijkstras as dgn

# depthMat = np.load('/home/clea/Desktop/TOF_person.npy')
# iters = 1
# centroid = []
# use_centroid = False


def generateKeypoints(depthMat, iters=1, draw_trails=False, centroid=[], use_centroid=False):
	'''
	---Parameters---
	depth image
	---Returns---
	extrema : set of (u,v) points
	allTrails :
	distance_map :
	'''
	objSize = depthMat.shape

	# Get discrete form of position/depth matrix
	if len(objSize) == 3:
		depthMatMin = depthMat[depthMat[:,:,2]>0, 2].min()
		depthMatMax = depthMat[:,:,2].max()
		depthMatDiscrete = np.ascontiguousarray(np.array((1-(depthMat[:,:,2]-depthMatMin)/(depthMatMax-depthMatMin))*32000, dtype=np.uint16))
	else:
		depthMatMin = depthMat[depthMat>0].min()# + 100
		depthMatMax = depthMat[depthMat>0].max()
		# depthMatDiscrete = np.ascontiguousarray(np.array((1-(depthMat-depthMatMin)/(depthMatMax-depthMatMin))*32000, dtype=np.uint16))
		depthMatDiscrete = np.ascontiguousarray(np.array(((depthMat.astype(np.float)-depthMatMin)/(depthMatMax-depthMatMin).astype(np.float))*32000, dtype=np.uint16))

	# mask = np.all(depthMat != 0, 2)
	mask = depthMat > 0
	depthMatDiscrete *= mask

	# Get starting position
	if centroid == []:
		if len(objSize) == 3:
			com = np.array(nd.center_of_mass(depthMat[:,:,2]), dtype=int)
		else:
			com = np.array(nd.center_of_mass(depthMat), dtype=int)
		startingPos = np.array([com[0], com[1]], dtype=np.int16)
	else:
		startingPos = np.array(centroid, dtype=np.int16)
	extrema = []
	# extrema = [startingPos]

	trail = []
	allTrails = []
	combinedTrail = set()
	
	# Run dijkstras
	for i in xrange(iters):

		distance_map = np.zeros([objSize[0],objSize[1]], dtype=np.uint16)+32000	
		# Set starting point as zero weight
		distance_map[startingPos[0], startingPos[1]] = 0

		# Set which pixels are in/out of bounds
		visited_map = np.zeros_like(distance_map, dtype=np.uint8)
		visited_map[-mask] = 255

		# Set previous trails as zero distance
		for j in combinedTrail:
			distance_map[j[0], j[1]] = 0

		# Dijkstras!
		trail = dgn.graph_dijkstras(distance_map, visited_map, depthMatDiscrete.astype(np.uint16), startingPos)
		distance_map *= mask * (distance_map<32000)

		# Add/merge trails		
		allTrails.append(trail)
		for j in trail:
			if j[0] > 0 and j[1] > 0:
				combinedTrail.add((j[0], j[1]))

		extrema.append(np.array([trail[0][0], trail[0][1]], dtype=np.int16))
		
		# Get new starting position
		if use_centroid:
			startingPos = extrema[0]
		else:
			startingPos = np.array([extrema[-1][0], extrema[-1][1]], dtype=np.int16)

		# print startingPos

	# Create a union of all trails
	if draw_trails:
		if allTrails != []:
			for trails_i in allTrails:
				for i in trails_i:
					if i[0] > 0 and i[1] > 0:
						distance_map[i[0], i[1]] = distance_map.max()

		# Show trail on image
		for i in extrema:
			distance_map[i[0]-3:i[0]+4, i[1]-3:i[1]+4] = distance_map.max()



	return extrema, allTrails, distance_map




def drawTrail(im, trail):
	for i,j in trail:
		im[i,j] = 255
	return im