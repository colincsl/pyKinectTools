
''' 
Calculate normals of a depth map.

This code is slow! Make faster by using cython and/or gpu.

bleh, do efficient gradient method

Colin Lea
pyKinectTools
2012
'''

import numpy as np
from pyKinectTools.utils.DepthUtils import *

def calcNormals(posMat):
	mask = posMat[:,:,2] > 0
	gxx, gxy = np.gradient(posMat[:,:,0], 1)
	gyx, gyy = np.gradient(posMat[:,:,1], 1)
	gzx, gzy = np.gradient(posMat[:,:,2], 1)

	xVecs = np.dstack([gxx, gyx, gzx])
	# xVecs[xVecs > 100] = 0
	norms = np.sqrt(np.sum(xVecs**2, 2))
	xVecs /= norms[:,:, np.newaxis]
	
	yVecs = np.dstack([gxy, gyy, gzy])
	# yVecs[yVecs > 100] = 0
	norms = np.sqrt(np.sum(yVecs**2, 2))
	yVecs /= norms[:,:, np.newaxis]

	zVecs = np.cross(xVecs, yVecs)
	zVecs[zVecs > 1] = 0

def calculateNormals(posMat, radius=3):
	# print posMat
	assert len(posMat.shape)== 3, "Input should be a posMat"
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
	
	inds = np.nonzero(posMat[coordsStart[0]:coordsEnd[0], coordsStart[1]:coordsEnd[1], 2])
	inds = [inds[0] + coordsStart[0], inds[1] + coordsStart[0]]
	xyzFloor = posMat[inds[1], inds[0]]
	meanFloor = xyzFloor.mean(0)
	xyzFloor -= meanFloor
	U, _, vT = np.linalg.svd(xyzFloor, full_matrices=False)

	return vT


# normals = calculateNormals(posMat, 5)

# ''' find floor '''
# if 0:
# 	im = normals[:,:,2] < 0
# 	# Finds the normal of the largest segment
# 	labels = nd.label(im)
# 	objs = nd.find_objects(labels[0])#, labels[1])
# 	pxCounts = np.array([np.sum(labels[0][x]==i+1) for x,i in zip(objs, range(labels[1]))])
# 	largestSurfaceIndex = np.argmax(pxCounts[1:])+1
# 	inds = np.nonzero(labels[0]==largestSurfaceIndex+1)








## Visualization
# figure(1)
# plot(xyzNew[0], xyzNew[1], '.')
# xlabel('x'); ylabel('y'); axis('equal')
# figure(2)
# plot(xyzNew[0], xyzNew[2], '.')
# xlabel('x'); ylabel('z'); axis('equal')
# figure(3)
# plot(xyzNew[1], xyzNew[2], '.')
# xlabel('y'); ylabel('z'); axis('equal')
# Put in image

# put in image form


