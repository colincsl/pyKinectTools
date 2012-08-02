import numpy as np
import scipy.ndimage as nd
import time

from pyKinectTools.algs.dijkstras import dijkstrasGraph

def generateKeypoints(posMat, xyz, iters=1, viz=0):

	# timeStart = time.time()
	objTmp = posMat
	obj2Size = np.shape(objTmp)

	mask = np.all(posMat != 0, 2)
	mask_erode = nd.binary_erosion(mask, iterations=2)

	com = np.array(nd.center_of_mass(posMat[:,:,2]), dtype=int)
	# tmp1 = np.nonzero(mask>0)
	# t = np.argmax(tmp1[0])	
	# current = [tmp1[0][t]-5, tmp1[1][t]]	
	current = [com[0], com[1]]

	trail = []
	allTrails = []
	singleTrail = set()

	# Find gradient
	dists2 = np.empty([obj2Size[0]-2,obj2Size[1]-2,4], dtype=np.int16)
	dists2[:,:,0] = 1*np.sum(np.abs(objTmp[1:-1, 1:-1] - objTmp[0:-2, 1:-1]), 2)#up
	dists2[:,:,1] = 1*np.sum(np.abs(objTmp[1:-1, 1:-1] - objTmp[2:, 1:-1]), 2)#down
	dists2[:,:,2] = 1*np.sum(np.abs(objTmp[1:-1, 1:-1] - objTmp[1:-1, 2:]), 2)#right
	dists2[:,:,3] = 1*np.sum(np.abs(objTmp[1:-1, 1:-1] - objTmp[1:-1, 0:-2]), 2)#left
	dists2[-mask_erode[1:-1,1:-1]] = 32000
	dists2[dists2>100] = 32000
	# dists2 = np.abs(dists2)

	# dists2copy = deepcopy(dists2)

	# extrema = []
	extrema = [current]
	t2 = time.time()
	for i in xrange(iters):
		dists2Tot = np.zeros([obj2Size[0],obj2Size[1]], dtype=np.int16)+32000		
		maxDists = np.max(dists2, 2)

		# distThresh = 50
		# outline = np.nonzero(maxDists>distThresh)
		# mask[outline[0]+1, outline[1]+1] = 0

		# dists2Tot[dists2Tot > 0] = 32000
		# dists2Tot[-mask] = 15000
		# dists2Tot[-mask_erode] = 1#15000
		dists2Tot[current[0], current[1]] = 0

		visitMat = np.zeros_like(dists2Tot, dtype=np.uint8)
		visitMat[-mask_erode] = 255

		for j in singleTrail:
			dists2Tot[j[0], j[1]] = 0
			dists2[j[0],j[1],:] = 0
			if 0 < (j[0]+1) < obj2Size[0]-2 and 0 < (j[1]+1) < obj2Size[1]-2:
				dists2[j[0]+1,j[1],0] = 0
				dists2[j[0]-1,j[1],1] = 0
				dists2[j[0],j[1]+1,2] = 0
				dists2[j[0],j[1]-1,3] = 0

		trail = dijkstrasGraph.graphDijkstras(dists2Tot, visitMat, dists2, current)

		# dists2Tot *= mask_erode
		dists2Tot *= (dists2Tot<32000)
		# dists2Tot *= mask
		# dists2Tot[1:,:] *= ((dists2Tot[1:,:] - dists2Tot[0:-1,:]) < 1000)
		# dists2Tot[:-1,:] *= ((dists2Tot[:-1,:] - dists2Tot[1:,:]) < 1000)
		# dists2Tot[:,1:] *= ((dists2Tot[:,1:] - dists2Tot[:,:-1]) < 1000)
		# dists2Tot[:,:-1] *= ((dists2Tot[:,:-1] - dists2Tot[:,1:]) < 1000)

		# maxInd = np.argmax(dists2Tot*(dists2Tot<30000))
		# maxInd = np.unravel_index(maxInd, dists2Tot.shape)
		# maxInd = (trail[-1][0],trail[-1][1])
		maxInd = (trail[0][0],trail[0][1])

		allTrails.append(trail)
		for j in trail:
			if j[0] > 0 and j[1] > 0:
				singleTrail.add((j[0], j[1]))

		# extrema.append(trail[-1])
		# current = [extrema[-1]]
		extrema.append([maxInd[0], maxInd[1]])
		current = [maxInd[0], maxInd[1]]
		# print current
	# print "t1: ", time.time() - t2

	trailLens = []
	for i in allTrails:
		trailLens.append(len(i))
	maxLen = np.argmax(trailLens)

	if allTrails != []:
		for trails_i in allTrails:
			for i in trails_i:
				if i[0] > 0 and i[1] > 0:
					dists2Tot[i[0], i[1]] = 1000
	for i in extrema:
		dists2Tot[i[0]-3:i[0]+4, i[1]-3:i[1]+4] = 1000#799

	# dists2Tot[extrema[maxLen][0]-3:extrema[maxLen][0]+4, extrema[maxLen][1]-3:extrema[maxLen][1]+4] = 400

	if viz:
		figure(1); imshow(dists2Tot*(dists2Tot <= 32000)*(dists2Tot > 0))


	return extrema, trail, dists2Tot


