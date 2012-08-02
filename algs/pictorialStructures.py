import numpy as np
from pyKinectTools.algs.skeletonBeliefPropagation import *
from pyKinectTools.algs.graphAlgs import *
from pyKinectTools.algs.orientationEstimate import *
from copy import deepcopy
import pyKinectTools.algs.neighborSuperpixels.neighborSuperpixels as nsp
from matplotlib.pyplot import text

import sys
sys.path.append('/Users/colin/libs/visionTools/slic-python/')
import slic


# class PictorialStructures:

# 	#  Learn Prior (use negative images)
# 	#  Sample from posterior
# 	#  SVM
# 	#  
# 	# Steps:
# 	# 	1) Calculate HOG Features OR use regions
# 	# 	2) 
# 	# 	3) Inference: Belief Propogation


# 	def __init__():
# 		pass

# 	def setPotentialPoses(self, poses):
# 		pass

# 	def setLabels(self, labels):
# 		pass

# 	def Score(Im, z):
# 		# z=configuration
# 		sum = 0
# 		for each part:
# 			sum += w_part * appearanceFcn

# 		for each set of edges
# 			sum += w_ij * deformation term


		# E = sum(how well does this part fit + how well does it fit it's neighbors)

	# torso->shoulder, torso->shoulder, sh->arm, sh->arm
	# np.sqrt(np.sum((regionXYZ[6]-regionXYZ[3])**2,0))
	# jointLengths = [200,200,50,50]
	# jointStart = [0,0,1,2]
	# jointEnd = [1,2,3,3]
	# bendParam = 10.0
	# Scores = []
	# Config = []
	# # for b in range(1, regionCount):
	# if 1:
	# 	b=3
	# 	score_i = 0
	# 	currPart = b
	# 	config_i = [b]
	# 	for l in range(len(jointLengths)):
	# 		partCosts = np.abs(distMat[config_i[jointStart[l]],:]-jointLengths[l])
	# 		partCosts += bendParam*bendMat[config_i[jointStart[l]],nextPart]
	# 		nextPart = np.argmin(partCosts)
	# 		score_i += partCosts[nextPart]
	# 		config_i.append(nextPart)
	# 		# print "----"
	# 		# print np.abs(distMat[config_i[jointStart[l]],nextPart]-jointLengths[l])
	# 		# print bendParam*bendMat[config_i[jointStart[l]],nextPart]		
	# 	Scores.append(score_i)
	# 	Config.append(config_i)

	# argmin_ = np.argmin(Scores)
	# print argmin_, Scores[argmin_]
	# print Config[argmin_]


def pictorialScores(regionXYZ,xyz, edgeDict):

	regionCount = len(regionXYZ)-1
	forwardVec, frame = roughOrientationEstimate(xyz)
	regionMean = np.mean(regionXYZ[1:],0)
	forwardRegionXYZ = np.asarray(np.asmatrix(frame)*np.asmatrix(regionXYZ-regionMean).T).T
	forwardRegionXYZ += regionMean
	
	relDists = np.ones([regionCount+1,regionCount+1,3])*np.inf	
	for i in range(1,regionCount+1):
		relDists[i,1:] = (forwardRegionXYZ[1:] - forwardRegionXYZ[i])

	for i in range(3):
		np.fill_diagonal(relDists[:,:,i], np.inf)

	distMat = UnstructuredDijkstras(regionXYZ, edgeDict)

	# --- Structure params ----
	HEAD=0;
	L_SHOUL=1; L_ELBOW=2; L_HAND=3;
	R_SHOUL=4; R_ELBOW=5; R_HAND=6;

	jointType = ['xy','r','r',
				 'xy','r','r']
	jointPos = np.array([[-200,-50], 100, 150,
						 [200,-50],  100, 150])
	jointStart = [HEAD,L_SHOUL,L_ELBOW, HEAD,R_SHOUL,R_ELBOW]
	jointEnd = [L_SHOUL,L_ELBOW,L_HAND, R_SHOUL,R_ELBOW,R_HAND]
	# --- /Structure params ----

	Scores = []
	Config = []
	for b in range(1, regionCount):
		score_i = 0
		currPart = b
		config_i = [b]
		for l in range(len(jointPos)):
			if jointType[l] == 'xy':
				partCosts = np.sum(np.abs(relDists[config_i[jointStart[l]],:,0:3:2] - jointPos[l]), 1)
			elif jointType[l] == 'r':
				partCosts = np.abs(distMat[config_i[jointStart[l]],:] - jointPos[l])
			nextPart = np.argmin(partCosts)
			score_i += partCosts[nextPart]
			config_i.append(nextPart)
		Scores.append(score_i)
		Config.append(config_i)

	argmin_ = np.argmin(Scores)
	# print argmin_, Scores[argmin_]
	# print Config[argmin_]

	return Config[argmin_]



# fig = figure(1)
# ax = Axes3D(fig)
# xlabel('X'); ylabel('Y'); axis('equal')
# ax.plot(-forwardRegionXYZ[:,0],forwardRegionXYZ[:,1],forwardRegionXYZ[:,2],'g.')

# figure(1)
# plot(regionXYZ[:,1],regionXYZ[:,0], 'b.')
# plot(-forwardRegionXYZ[:,0],forwardRegionXYZ[:,1], 'g.')


		# for b2 in range(1, regionCount):




def regionGraph(posMat):

	im8bit = deepcopy(posMat)
	mask_erode = posMat[:,:,2]>0
	for i in range(3):
		im8bit[:,:,i][im8bit[:,:,i]!=0] -= im8bit[:,:,i][im8bit[:,:,i]!=0].min()
		im8bit[:,:,i] /= im8bit[:,:,i].max()/256
	im8bit = np.array(im8bit, dtype=np.uint8)
	# im8bit = im8bit[:,:,2]
	im4d = np.dstack([mask_erode, im8bit])
	# im4d = np.dstack([mask_erode, im8bit, im8bit, im8bit])
	# im4d = np.dstack([mask_erode, dists2Tot, dists2Tot, dists2Tot])
	# im4d = np.dstack([mask_erode, im8bit, dists2Tot, mask_erode])
	regions = slic.slic_n(np.array(im4d, dtype=np.uint8), 50,10)#2
	# regions = slic.slic_s(np.array(im4d, dtype=uint8), 550,3)
	regions *= mask_erode
	# imshow(regions)

	avgColor = np.zeros([regions.shape[0],regions.shape[1],3])

	regionCount = regions.max()
	regionLabels = [[0]]
	goodRegions = 0
	bodyMean = np.array([posMat[mask_erode,0].mean(),posMat[mask_erode,1].mean(),posMat[mask_erode,2].mean()])
	for i in range(1, regionCount+2):
		if np.sum(np.equal(regions,i)) < 100:
			regions[regions==i] = 0
		else:
			if 1: #if using x/y/z
				meanPos = posMat[regions==i,:].mean(0)
			if 0: # If using distance map
				meanPos = np.array([posMat[regions==i,:].mean(0)[0],
									posMat[regions==i,:].mean(0)[1],
									# posMat[regions==i,:].mean(0)[2],
									(dists2Tot[regions==i].mean())])		
			if 0: # If using depth only
				meanPos = np.array([(np.nonzero(regions==i)[0].mean()),
							(np.nonzero(regions==i)[1].mean()),
							(im8bit[regions==i].mean())])
			avgColor[regions==i,:] = meanPos - bodyMean
			if not np.isnan(meanPos[0]) and meanPos[0] != 0.0:
				tmp = np.nonzero(regions==i)
				argPos = [int(tmp[0].mean()),int(tmp[1].mean())]
				regionLabels.append([i, meanPos-bodyMean, argPos])
				goodRegions += 1
				regions[regions==i] = goodRegions
				# print np.sum(np.equal(regions,i))
			else:
				regions[regions==i] = 0
	regionCount = regions.max()



	allEdges = nsp.getNeighborEdges(np.ascontiguousarray(regions, dtype=np.uint8))

	edges = []
	for i in allEdges:
		if i[0] != 0 and i[1] != 0:
			if i not in edges:
				edges.append(i)
			if [i[1],i[0]] not in edges:
				edges.append([i[1],i[0]])

	edgeDict = edgeList2Dict(edges)


	regionXYZ = ([x[1] for x in regionLabels if x[0] != 0])
	regionXYZ.insert(0,[0,0,0])
	regionPos = ([x[2] for x in regionLabels if x[0] != 0])	
	regionPos.insert(0,[0,0])


	# distMat = UnstructuredDijkstras(regionXYZ, edgeDict)
	distMat, bendMat = UnstructuredDijkstrasAndBend(regionXYZ, edgeDict)

	mstEdges, edgeDict2 = MinimumSpanningTree(distMat[1:,1:])


	return regions, regionXYZ, regionLabels, edgeDict


def labelGraphImage(regionLabels):
	regionCount = len(regionLabels)-1
	for i in range(1,regionCount+1):
		pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
		# cv2.circle(imLines, pt1, radius=0, color=50, thickness=3)
		# cv2.circle(imLines, pt1, radius=0, color=distMat[1, i]*255.0/maxDist, thickness=3)
		text(pt1[0]+2, pt1[1], str(i))



if 0:
	# maxDist = distMat[distMat<np.inf].max()
	maxDist = distMat[1,2:].max()
	minDist = distMat[1,1:].min()
	# Draw lines between nodes
	imLines = deepcopy(regions)
	removeEdges = []
	for i, ind in zip(edges, range(len(edges))):
		i1 = i[0]
		i2 = i[1]
	for i, ind in zip(mstEdges, range(len(edges))):	
		i1 = i[0]+1
		i2 = i[1]+1
		pt1 = (regionLabels[i1][2][1],regionLabels[i1][2][0])
		pt2 = (regionLabels[i2][2][1],regionLabels[i2][2][0])
		cv2.line(imLines, pt1, pt2, 40)
		# cv2.line(imLines, pt1, pt2, 255.0/maxDist*distMat[18, edges[i[1]]])
		# cv2.line(imLines, pt1, pt2, 255.0/maxDist*distMat[edges[i[0]], edges[i[1]]])
	for i in range(1,regionCount+1):
		pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
		# cv2.circle(imLines, pt1, radius=0, color=50, thickness=3)
		# cv2.circle(imLines, pt1, radius=0, color=distMat[1, i]*255.0/maxDist, thickness=3)
		
		text(pt1[0]+2, pt1[1], str(i))

	imshow(imLines)





	# CRF?
	# Energies: length + bend angle (resultant vector?)







	# # Test on ICU data
	# image_argb = dstack([d1c, d1c, d1c, d1c])
	# image_argb = dstack([m1, m1, m1, m1])
	# # region_labels = slic.slic_s(image_argb, 10000, 1)

	# image_argb = dstack([diffDraw1,diffDraw1,diffDraw1,diffDraw1])
	# region_labels = slic.slic_n(image_argb, 100, 0)
	# slic.contours(image_argb, region_labels, 1)
	# plt.imshow(image_argb[:, :, 0])


	# regions = slic.slic_n(np.array(np.dstack([im,im[:,:,2]]), dtype=uint8), 50,10)

	# i=3
	# # im8bit = np.array(1.0*imgs2[i]/imgs2[i].max()*256.0)
	# im8bit = im*(im<150)*(im>50)
	# # im8bit = im
	# im4d = np.dstack([im8bit>0, im8bit, im8bit, im8bit])
	# regions = slic.slic_n(np.array(im4d, dtype=uint8), 100,5)
	# regions *= (im8bit>0)
	# regions2 = slic.slic_n(np.array(im4d, dtype=uint8), 50,5)
	# regions2 *= (im8bit>0)
	# regions3 = slic.slic_n(np.array(im4d, dtype=uint8), 20,1)
	# regions3 *= (im8bit>0)
	# # regions = slic.slic_n(np.array(im4d, dtype=uint8), 30,5)
	# imshow(regions)
	# -----

	# dists2Tot[dists2Tot>1000] = 1000

	# im8bit = (d[objects[ind]]*mask_erode)
	# im8bit = im8bit / np.ceil(im8bit.max()/256.0)
	im8bit = deepcopy(posMat)
	for i in range(3):
		im8bit[:,:,i][im8bit[:,:,i]!=0] -= im8bit[:,:,i][im8bit[:,:,i]!=0].min()
		im8bit[:,:,i] /= im8bit[:,:,i].max()/256
	im8bit = np.array(im8bit, dtype=uint8)
	# im8bit = im8bit[:,:,2]
	im4d = np.dstack([mask_erode, im8bit])
	# im4d = np.dstack([mask_erode, im8bit, im8bit, im8bit])
	# im4d = np.dstack([mask_erode, dists2Tot, dists2Tot, dists2Tot])
	# im4d = np.dstack([mask_erode, im8bit, dists2Tot, mask_erode])
	regions = slic.slic_n(np.array(im4d, dtype=uint8), 50,10)#2
	# regions = slic.slic_s(np.array(im4d, dtype=uint8), 550,3)
	regions *= mask_erode
	imshow(regions)

	avgColor = np.zeros([regions.shape[0],regions.shape[1],3])
	# avgColor = np.zeros([regions.shape[0],regions.shape[1],4])

	regionCount = regions.max()
	regionLabels = [[0]]
	goodRegions = 0
	bodyMean = np.array([posMat[mask_erode,0].mean(),posMat[mask_erode,1].mean(),posMat[mask_erode,2].mean()])
	for i in range(1, regionCount+2):
		if np.sum(np.equal(regions,i)) < 50:
			regions[regions==i] = 0
		else:
			if 1: #if using x/y/z
				meanPos = posMat[regions==i,:].mean(0)
			if 0: # If using distance map
				meanPos = np.array([posMat[regions==i,:].mean(0)[0],
									posMat[regions==i,:].mean(0)[1],
									# posMat[regions==i,:].mean(0)[2],
									(dists2Tot[regions==i].mean())])		
			if 0: # If using depth only
				meanPos = np.array([(np.nonzero(regions==i)[0].mean()),
							(np.nonzero(regions==i)[1].mean()),
							(im8bit[regions==i].mean())])
			avgColor[regions==i,:] = meanPos - bodyMean
			if not np.isnan(meanPos[0]) and meanPos[0] != 0.0:
				tmp = np.nonzero(regions==i)
				argPos = [int(tmp[0].mean()),int(tmp[1].mean())]
				regionLabels.append([i, meanPos-bodyMean, argPos])
				goodRegions += 1
				regions[regions==i] = goodRegions
				# print np.sum(np.equal(regions,i))
			else:
				regions[regions==i] = 0
	regionCount = regions.max()

	#Reindex
	regionCount = len(regionLabels)
	for lab, j in zip(regionLabels, range(regionCount)):
		lab.append(j)
		# mapRegionToIndex.append(lab[0])

	# (Euclidan) Distance matrix
	distMatrix = np.zeros([regionCount, regionCount])
	for i_data,i_lab in zip(regionLabels, range(regionCount)):
		for j_data,j_lab in zip(regionLabels, range(regionCount)):
			if i_lab <= j_lab:
				# distMatrix[i_lab,j_lab] = np.sqrt(((i_data[1][0]-j_data[1][0])**2)+((i_data[1][1]-j_data[1][1])**2)+.5*((i_data[1][2]-j_data[1][2])**2))
				distMatrix[i_lab,j_lab] = np.sqrt(np.sum((i_data[1]-j_data[1])**2))
	distMatrix = np.maximum(distMatrix, distMatrix.T)
	distMatrix += 1000*eye(regionCount)
	# distMatrix[distMatrix > 400] = 1000
	edges = distMatrix.argmin(0)

	if 0:
		''' Draw edges based on closest node '''
		imLines = deepcopy(regions)
		for i in range(regionCount):
			pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
			cv2.circle(imLines, pt1, radius=0, color=125, thickness=3)

		for i in range(regionCount):
			pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
			pt2 = (regionLabels[edges[i]][2][1],regionLabels[edges[i]][2][0])
			cv2.line(imLines, pt1, pt2, 100)

	mstEdges, edgeDict = MinimumSpanningTree(distMatrix)

	# ''' Refine MST '''
	# edgeDict, deletedInds = PruneEdges(edgeDict, maxLength=2)

	# for i in deletedInds[-1::-1]:
	# 	del regionLabels[i]

	# #Reindex
	# regionCount = len(regionLabels)
	# for lab, j in zip(regionLabels, range(regionCount)):
	# 	lab.append(j)
	# 	# mapRegionToIndex.append(lab[0])

	# # (Euclidan) Distance matrix
	# distMatrix = np.zeros([regionCount, regionCount])
	# for i_data,i_lab in zip(regionLabels, range(regionCount)):
	# 	for j_data,j_lab in zip(regionLabels, range(regionCount)):
	# 		if i_lab <= j_lab:
	# 			# distMatrix[i_lab,j_lab] = np.sqrt(((i_data[1][0]-j_data[1][0])**2)+((i_data[1][1]-j_data[1][1])**2)+.5*((i_data[1][2]-j_data[1][2])**2))
	# 			distMatrix[i_lab,j_lab] = np.sqrt(np.sum((i_data[1]-j_data[1])**2))
	# distMatrix = np.maximum(distMatrix, distMatrix.T)
	# distMatrix += 1000*eye(regionCount)
	# edges = distMatrix.argmin(0)

	# mstEdges, edgeDict = MinimumSpanningTree(distMatrix)



	# figure(1); imshow(objTmp[:,:,2])

	''' Draw edges based on minimum spanning tree '''
	imLines = deepcopy(regions)
	for i in range(1,regionCount):
		pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
		cv2.circle(imLines, pt1, radius=0, color=125, thickness=3)
	# mstEdges = np.array(mstEdges) + 1
	# Draw line for all edges
	if 1:
		for i in range(len(mstEdges)):
			try:
				pt1 = (regionLabels[mstEdges[i][0]][2][1],regionLabels[mstEdges[i][0]][2][0])
				pt2 = (regionLabels[mstEdges[i][1]][2][1],regionLabels[mstEdges[i][1]][2][0])
				cv2.line(imLines, pt1, pt2, 100)
			except:
				pass
	figure(2); imshow(imLines)

	''' Draw line between all core nodes '''

	# Draw circles
	imLines = deepcopy(regions)
	for i in range(1,regionCount):
		pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
		cv2.circle(imLines, pt1, radius=0, color=125, thickness=3)

	leafPaths = GetLeafLengths(edgeDict)
	leafLengths = [len(x) for x in leafPaths]
	core = [x for x in edgeDict.keys() if len(edgeDict[x]) > 2]
	branchesSet = set()
	for i in leafPaths:
		for j in i:
			branchesSet.add(j)
	core = np.sort(list(set(range(regionCount)).difference(branchesSet)))
	# core = [x for x in edgeDict.keys() if len(edgeDict[x]) > 2]
	for i in range(len(core)-1):
		pt1 = (regionLabels[core[i]][2][1], regionLabels[core[i]][2][0])
		pt2 = (regionLabels[core[i+1]][2][1],regionLabels[core[i+1]][2][0])
		cv2.line(imLines, pt1, pt2, 150)


	# Draw line for all leafs
	for i in range(len(leafPaths)):
		if len(leafPaths[i]) > 3:
			color = 125
		else:
			color = 100
		for j in range(len(leafPaths[i])-1):
			pt1 = (regionLabels[leafPaths[i][j]][2][1],regionLabels[leafPaths[i][j]][2][0])
			pt2 = (regionLabels[leafPaths[i][j+1]][2][1],regionLabels[leafPaths[i][j+1]][2][0])
			cv2.line(imLines, pt1, pt2, color)


	#Draw head and hands
	pt1 = (regionLabels[core[0]][2][1],regionLabels[core[0]][2][0])
	cv2.circle(imLines, pt1, radius=10, color=150, thickness=1)

	for i in xrange(len(leafLengths)):
		if leafLengths[i] >= 4:
			pt1 = (regionLabels[leafPaths[i][0]][2][1],regionLabels[leafPaths[i][0]][2][0])
			cv2.circle(imLines, pt1, radius=10, color=125, thickness=1)



	figure(3); imshow(imLines)






	if 1:
		imLines = deepcopy(regions)
		imLines[imLines>0] = 20
		for i in range(len(mstEdges)):
			pt1 = (regionLabels[mstEdges[i][0]][2][1],regionLabels[mstEdges[i][0]][2][0])
			pt2 = (regionLabels[mstEdges[i][1]][2][1],regionLabels[mstEdges[i][1]][2][0])
			cv2.line(imLines, pt1, pt2, 3)


		# head, body, arm, legs
		# potentialPoses = [np.array([[500, 30, 0], [50, 44, -27], [-18, -150, 25]]),
		# 				  np.array([[500, 30, 0], [107, 44, 0], [-18, 150, 25]]),
		# 				  np.array([[0, 30, 0], [107, 44, 0], [-18, -150, 25]]),
		# 				  np.array([[200, 30, 0], [107, 144, 0], [-18, -150, 25]])]
		# 				  np.array([[500, 0, -25], [-107, 144, 100], [-18, -150, 25]])]

		potentialPoses = [np.array([regionLabels[3][1], regionLabels[27][1], regionLabels[24][1],regionLabels[55][1]]),
						  np.array([regionLabels[7][1], regionLabels[30][1], regionLabels[22][1],regionLabels[53][1]]),
						  np.array([regionLabels[5][1], regionLabels[22][1], regionLabels[29][1],regionLabels[54][1]]),
						  np.array([regionLabels[0][1], regionLabels[23][1], regionLabels[24][1],regionLabels[55][1]])]
		potentialLabels = [np.array([regionLabels[3][2], regionLabels[27][2], regionLabels[24][2],regionLabels[55][2]]),
						  np.array([regionLabels[7][2], regionLabels[30][2], regionLabels[22][2],regionLabels[53][2]]),
						  np.array([regionLabels[5][2], regionLabels[22][2], regionLabels[29][2],regionLabels[54][2]]),
						  np.array([regionLabels[0][2], regionLabels[23][2], regionLabels[24][2],regionLabels[55][2]])]  

		# transitionMatrix = np.matrix([[.1,.45, .45],[.45,.1, .45],[.45,.45,.1]])
		# transitionMatrix = np.matrix([[.5,.25, .25],[.25,.5, .25],[.25,.25,.5]])
		# transitionMatrix = np.matrix([[.9,.05, .05],[.05,.9, .05],[.05,.05,.9]])

		transitionMatrix = np.matrix([[.55,.15, .15, .15],[.15,.55, .15, .15],[.15,.15,.55, .15],[.15,.15,.15,.55]])
		# transitionMatrix = np.matrix([[.7,.1, .1, .1],[.1,.7, .1, .1],[.1,.1,.7, .1],[.1,.1,.1,.7]])	
		# transitionMatrix = np.matrix([[1,.0, .0, .0],[.0,1, .0, .0],[.0,.0,1, .0],[.0,.0,.0,1]])	
		# transitionMatrix = np.matrix([[0,1.0,1.0,1.0],[1.0,.0,1.0,1.0],[1.0,1.0,.0,1.0],[1.0,1.0,1.0,.0]])	
		# transitionMatrix = np.matrix([[.0,.0, .0, .0],[.0,0, .0, .0],[.0,.0,0, .0],[.0,.0,.0,0]])	


		rootNodeInd = core[int(len(core)/2)]
		rootNode = Node(index_=rootNodeInd, children_=edgeDict[rootNodeInd], pos_=regionLabels[rootNodeInd][1])

		beliefs = []
		ims = []
		for guessPose,i in zip(potentialPoses, range(len(potentialPoses))):
			print "-----"
			# print guessPose
			t1 = time.time()
			rootNode.calcAll(guessPose)
			print "Time:", time.time() - t1

			beliefs.append(rootNode.calcTotalBelief())
			print beliefs[-1]
			rootNode.drawAll()

			ims.append(deepcopy(imLines))
			pts = potentialLabels[i]
			for j,j_i in zip(pts, range(len(pts))):
				print j
				cv2.circle(ims[-1], (j[1], j[0]), radius=15, color=20*j_i+10, thickness=2)
			subplot(1,4,i+1)
			imshow(ims[-1])



		print "Best pose:", np.argmax(beliefs)
		subplot(1,4,np.argmax(beliefs)+1)
		xlabel("**Best**")


		# imshow(imLines)

