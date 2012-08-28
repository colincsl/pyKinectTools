import numpy as np
import scipy.ndimage as nd
import sys
# import gc

# from pyKinectTools.algs.dijkstras import dijkstrasGraph
from pyKinectTools.algs.Dijkstras import dijkstrasGraphNew as dgn

def generateKeypoints(depthMat, iters=1, centroid=[], use_centroid=False):

	# gc.disable()

	objSize = depthMat.shape

	# Get discrete form of position/depth matrix
	if len(objSize) == 3:
		depthMatMin = depthMat[depthMat[:,:,2]>0, 2].min()
		depthMatMax = depthMat[:,:,2].max()
		depthMatDiscrete = np.ascontiguousarray(np.array((1-(depthMat[:,:,2]-depthMatMin)/(depthMatMax-depthMatMin))*32000, dtype=np.uint16))
	else:
		depthMatMin = depthMat[depthMat>0].min()# + 100
		depthMatMax = depthMat.max()
		depthMatDiscrete = np.ascontiguousarray(np.array((1-(depthMat-depthMatMin)/(depthMatMax-depthMatMin))*32000, dtype=np.uint16))

	mask = np.all(depthMat != 0, 2)
	depthMatDiscrete *= mask

	# Get starting position
	if centroid == []:
		com = np.array(nd.center_of_mass(depthMat[:,:,2]), dtype=int)
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

		distsMat = np.zeros([objSize[0],objSize[1]], dtype=np.uint16)+32000	
		# Set starting point as zero weight
		distsMat[startingPos[0], startingPos[1]] = 0

		# Set which pixels are in/out of bounds
		visitMat = np.zeros_like(distsMat, dtype=np.uint8)
		visitMat[-mask] = 255

		# Set previous trails as zero distance
		for j in combinedTrail:
			distsMat[j[0], j[1]] = 0

		# Dijkstras!
		trail = dgn.graphDijkstras(distsMat, visitMat, depthMatDiscrete, startingPos)
		distsMat *= mask * (distsMat<32000)


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
	if allTrails != []:
		for trails_i in allTrails:
			for i in trails_i:
				if i[0] > 0 and i[1] > 0:
					distsMat[i[0], i[1]] = distsMat.max()

	# Show trail on image
	for i in extrema:
		distsMat[i[0]-3:i[0]+4, i[1]-3:i[1]+4] = distsMat.max()

	# gc.enable()

	return extrema, allTrails, distsMat


