
#python -m cProfile dijkstrasTest.pyx dijkProfile

import os, time, sys
import numpy as np
import cv, cv2
import scipy.ndimage as nd
import pdb
from math import floor
sys.path.append('/Users/colin/code/Kinect-Projects/activityRecognition/')
from icuReader import ICUReader
from peopleTracker import Tracker
from SkelPlay import *
from backgroundSubtract import *
from featureExtraction import *

# import pyximport
# pyximport.install()
import dijkstrasGraph


# def getExtrema(objects, labelInds, out, d, com, featureExt, ind):
	# objects = objects1
	# labelInds = labelInds1
	# out = out1
	# d=d1
	# com = com1
	# featureExt = featureExt1

saved = np.load('tmpPerson.npz')['arr_0'].tolist()
objects1 = saved['objects']; labelInds1=saved['labels']; out1=saved['out']; d1=saved['d']; com1=saved['com'];featureExt1=saved['features']
objects = objects1
labelInds = labelInds1
out = out1
d=d1
com = com1
featureExt = featureExt1
ind=1

extrema = []
mask = out[objects[ind]]==labelInds[ind]
mask_erode = nd.binary_erosion(out[objects[ind]]==labelInds[ind], iterations=5)

objTmp = np.array(d[objects[ind]])#, dtype=np.uint16)

obj2Size = np.shape(objTmp)
x = objects[ind][0].start # down
y = objects[ind][1].start # right
c = np.array([com[ind][0] - x, com[ind][1] - y])
current = [c[0], c[1]]
xyz = featureExt.xyz[ind]
trailSets = []

# distsMat = np.ones([obj2Size[0],obj2Size[1]], dtype=uint16)*np.inf

## Init
com_xyz = depth2world(np.array([[current[0]+x, current[1]+y, d[current[0]+x, current[1]+y]]]))[0]
dists = np.sqrt(np.sum((xyz-com_xyz)**2, 1))
inds = featureExt.indices[ind]

distsMat = np.zeros([obj2Size[0],obj2Size[1]], dtype=uint16)		
distsMat = ((-mask)*499)
distsMat[inds[0,:]-x, inds[1,:]-y] = dists 		

# objTmp = distsMat
dists2 = np.empty([obj2Size[0]-2,obj2Size[1]-2,4], dtype=int16)
dists2[:,:,0] = objTmp[1:-1, 1:-1] - objTmp[0:-2, 1:-1]#up
dists2[:,:,1] = objTmp[1:-1, 1:-1] - objTmp[2:, 1:-1]#down
dists2[:,:,2] = objTmp[1:-1, 1:-1] - objTmp[1:-1, 2:]#right
dists2[:,:,3] = objTmp[1:-1, 1:-1] - objTmp[1:-1, 0:-2]#left
dists2 = np.abs(dists2)

dists2Tot = np.zeros([obj2Size[0],obj2Size[1]], dtype=int16)+9999		
maxDists = np.max(dists2, 2)
distThresh = 20
outline = np.nonzero(maxDists>distThresh)
mask[outline[0]+1, outline[1]+1] = 0
mask = nd.binary_erosion(mask, iterations=2)


extrema.append(current)

time_ = time.time()
# if 1:
for k in xrange(20):
	com_xyz = depth2world(np.array([[current[0]+x, current[1]+y, d[current[0]+x, current[1]+y]]]))[0]
	dists = np.sqrt(np.sum((xyz-com_xyz)**2, 1))
	inds = featureExt.indices[ind]
	
	distsMat = np.zeros([obj2Size[0],obj2Size[1]], dtype=uint16)		
	distsMat = ((-mask)*499)
	distsMat[inds[0,:]-x, inds[1,:]-y] = dists 		

	# objTmp = distsMat
	objTmp *= mask

	# objTmp[current[0], current[1]] = 0
	# for t in trailSets:
	# 	for i in t:
	# 		objTmp[i[0], i[1]] = 0

	dists2 = np.empty([obj2Size[0]-2,obj2Size[1]-2,4], dtype=int16)
	dists2[:,:,0] = objTmp[1:-1, 1:-1] - objTmp[0:-2, 1:-1]#up
	dists2[:,:,1] = objTmp[1:-1, 1:-1] - objTmp[2:, 1:-1]#down
	dists2[:,:,2] = objTmp[1:-1, 1:-1] - objTmp[1:-1, 2:]#right
	dists2[:,:,3] = objTmp[1:-1, 1:-1] - objTmp[1:-1, 0:-2]#left
	dists2 = np.abs(dists2)

	dists2[current[0], current[1], :] = 0
	for t in trailSets:
		for i in t:
			dists2[i[0], i[1],:] = 0	

	m2 = -mask
	dists2[m2[1:-1, 1:-1]] = 15000
	# imshow(dists2[:,:,0]*(dists2[:,:,0]<1000))
	# imshow((dists2[:,:,0]<1000))


	dists2Tot = np.zeros([obj2Size[0],obj2Size[1]], dtype=int16)+9999		
	# maxDists = np.max(dists2, 2)
	# distThresh = 30
	# outline = np.nonzero(maxDists>distThresh)
	# mask[outline[0]+1, outline[1]+1] = 0
	# mask = nd.binary_erosion(mask, iterations=2)
	# dists2Tot[dists2Tot > 0] = 9999
	dists2Tot[-mask] = 15000
	# dists2Tot[-mask_erode] = 15000

	dists2Tot[current[0], current[1]] = 0

	for t in trailSets:
		for i in t:
			dists2Tot[i[0], i[1]] = 0

	visitMat = np.zeros_like(dists2Tot, dtype=uint8)
	visitMat[-mask] = 255

	# dists2Tot = dijkstrasGraph.dijkstras(dists2Tot, visitMat, dists2, current)
	trail = dijkstrasGraph.dijkstras(dists2Tot, visitMat, dists2, current)
	trailSets.append(trail)

	# dists2Tot *= mask_erode
	maxInd = (dists2Tot*(dists2Tot<9999)).argmax()
	maxInd = np.unravel_index(maxInd, dists2Tot.shape)

	extrema.append([maxInd[0], maxInd[1]])
	current = [maxInd[0], maxInd[1]]
print (time.time()-time_)

	# print trail

for i in extrema:
	dists2Tot[i[0]-3:i[0]+3, i[1]-3:i[1]+3] = 500
for t in trailSets:
	for i in t:
		dists2Tot[i[0], i[1]] = 500
imshow(dists2Tot*(dists2Tot <= 1000))
# imshow(dists2Tot<9999)



	# pdb.set_trace()
	# return extrema



# if __name__=="__main__":
# 	saved = np.load('tmpPerson.npz')['arr_0'].tolist()
# 	objects1 = saved['objects']; labelInds1=saved['labels']; out1=saved['out']; d1=saved['d']; com1=saved['com'];featureExt1=saved['features']

# 	extrema = getExtrema(objects1, labelInds1, out1, d1, com1, featureExt1, ind=1)




