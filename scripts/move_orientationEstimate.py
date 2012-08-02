from copy import deepcopy
import sys, os, time
import cv2, numpy as np
sys.path.append('/Users/colin/libs/visionTools/slic-python/')
import slic
import Image 
from pyKinectTools.algs.skeletonBeliefPropagation import *
from pyKinectTools.algs.graphAlgs import *
from pyKinectTools.algs.geodesicSkeleton import *
import pyKinectTools.algs.neighborSuperpixels.neighborSuperpixels as nsp

# saved = np.load('tmpPerson_close.npz')['arr_0'].tolist()
# saved = np.load('tmpPerson1.npz')['arr_0'].tolist()
# objects1 = saved['objects']; labelInds1=saved['labels']; out1=saved['out']; d1=saved['d']; com1=saved['com'];featureExt1=saved['features']; posMat=saved['posMat']; xyz=saved['xyz']
# posMat=saved['posMat']; xyz=saved['xyz']

time1 = time.time()
mask_erode = posMat[:,:,2]>0

im8bit = deepcopy(posMat)
for i in range(3):
	im8bit[:,:,i][im8bit[:,:,i]!=0] -= im8bit[:,:,i][im8bit[:,:,i]!=0].min()
	im8bit[:,:,i] /= im8bit[:,:,i].max()/256
im8bit = np.array(im8bit, dtype=uint8)
# im8bit = im8bit[:,:,2]
im4d = np.dstack([mask_erode, im8bit])
# im4d = np.dstack([mask_erode, im8bit, im8bit, im8bit])

# regions = slic.slic_n(np.array(im4d, dtype=uint8), 50,10)#2
regions = slic.slic_n(np.array(im4d, dtype=uint8), 50,10)#2
regions *= mask_erode

avgColor = np.zeros([regions.shape[0],regions.shape[1],3])

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


# ------------# ------------# ------------# ------------# ------------





regionXYZ = ([x[1] for x in regionLabels if x[0] != 0])
regionPos = ([x[2] for x in regionLabels if x[0] != 0])
regionXYZ.insert(0,[0,0,0])
regionPos.insert(0,[0,0])

''' Get upper and middle nodes '''
regionXYZ = np.array(regionXYZ)
midpoint = np.argmin(np.sum((regionXYZ[1:] - np.mean(regionXYZ[1:,0]))**2, 1))+1
uppermostPart = np.argmax(regionXYZ[1:,0])+1
upVec = regionXYZ[uppermostPart] - regionXYZ[midpoint]
upVec /= np.sum(upVec)

dists = np.sqrt(np.sum((regionXYZ[1:] - regionXYZ[midpoint])**2, 1))
tmpNodes = np.nonzero((100 < dists) * (dists < 250))[0]+1

upperMidpoint = (regionXYZ[midpoint] + regionXYZ[uppermostPart]) / 2
dists = np.sqrt(np.sum((regionXYZ[1:] - upperMidpoint)**2, 1))

## --- Head regions
headSize = 250
dists = np.sqrt(np.sum((regionXYZ[1:] - regionXYZ[uppermostPart])**2, 1))
headRegions = np.nonzero(dists<headSize)[0]+1
headXYZ = np.mean([regionXYZ[x] for x in headRegions], 0)
headPos = np.mean([regionPos[x] for x in headRegions], 0, dtype=int)
head = np.zeros_like(regions)
classifyBody = np.zeros_like(regions)
for i in headRegions:
	head += regions==i
	classifyBody[regions==i] = 1

headPtsXYZ = posMat[(head>0)]

## --- Torso
torsoXYZ = regionXYZ[midpoint]
torsoPos = regionPos[midpoint]
headXYZ = regionXYZ[uppermostPart]
neckXYZ = (torsoXYZ+headXYZ)/2
neckPos = np.sum([regionPos[midpoint],regionPos[uppermostPart]], 0)/2


tmpXYZ = posMat[(regions<=midpoint)*(regions>np.max(headRegions))]
tmpXYZ -= tmpXYZ.mean(0)
U,_,vT=svd((tmpXYZ-tmpXYZ.mean(0)), full_matrices=0)

tmpXYZ2 = xyz
tmpXYZ2 -= tmpXYZ2.mean(0)
_,_,vT2=svd(tmpXYZ2-tmpXYZ2.mean(0), full_matrices=0)

shoulderAxis = vT[2,:]
bodyAxis = vT2[2,:]
angleDiff = np.arccos(np.dot(bodyAxis,shoulderAxis))*180/np.pi


if viz:
	print "Angle diff:", angleDiff


if viz:
	figure(5)
	min_ = 1.0*posMat[:,:,1][posMat[:,:,2]>0].min()
	max_ = 1.0*posMat[:,:,1].max() - min_
	scatter(xyz[:,0],xyz[:,2], c=1.0*np.array([(posMat[:,:,1][posMat[:,:,2]>0]-min_)*1.0/max_], dtype=float))
	scatter(tmpXYZ[:,0],tmpXYZ[:,2], c='r')
	plot([0, 200*vT[2,0]], [0,200*vT[2,2]], c='g', linewidth=5)
	plot([0, 200*vT2[2,0]], [0,200*vT2[2,2]], c='k', linewidth=5)

tmpXYZ2 = (np.asmatrix(vT)*np.asmatrix(tmpXYZ.T)).T
tmpXYZ3 = (np.asmatrix(vT)*np.asmatrix(xyz.T)).T

shoulders = [np.argmin(tmpXYZ2[:,0]),np.argmax(tmpXYZ2[:,0])]
shoulderPx = []
for i in shoulders:
 	shoulderPx.append([np.nonzero((posMat[:,:,2]*(regions<=midpoint)*(regions>np.max(headRegions)))>0)[0][i], np.nonzero((posMat[:,:,2]*(regions<=midpoint)*(regions>np.max(headRegions)))>0)[1][i]])

if viz:
	figure(6)
	plot(tmpXYZ2[:,0],tmpXYZ2[:,1], 'b.')
	plot(tmpXYZ2[:,0],tmpXYZ2[:,2], 'g.')
	plot(tmpXYZ2[:,1],tmpXYZ2[:,2], 'r.')

# if viz:
if 1:
	im2 = deepcopy(regions)
	cv2.circle(im2, (headPos[1], headPos[0]), 10, 5) #Head
	cv2.circle(im2, (neckPos[1], neckPos[0]), 5, 5) #Neck
	cv2.circle(im2, (torsoPos[1], torsoPos[0]), 5, 5) #Torso
	for px in shoulderPx:
	 	cv2.circle(im2, (px[1], px[0]), 5, 5)
	# figure(8)
	imshow(im2)

if viz:

	# Neck:
	figure(6)
	# text(.5, .95, 'Upper body points', horizontalalignment='center') 
	subplot(2,2,1); axis('equal'); xlabel('X'); ylabel('Y')
	plot(-tmpXYZ2[:,0],-tmpXYZ2[:,1], 'b.')
	subplot(2,2,2); axis('equal'); xlabel('Z'); ylabel('Y')
	plot(-tmpXYZ2[:,2],-tmpXYZ2[:,1], 'b.')
	subplot(2,2,3); axis('equal'); xlabel('X'); ylabel('Z')
	plot(-tmpXYZ2[:,0],tmpXYZ2[:,2], 'b.')
	# Whole body:
	figure(7)
	subplot(2,2,1); axis('equal'); xlabel('X'); ylabel('Y')
	plot(tmpXYZ3[:,0],-tmpXYZ3[:,1], 'b.')
	subplot(2,2,2); axis('equal'); xlabel('Z'); ylabel('Y')
	plot(-tmpXYZ3[:,2],-tmpXYZ3[:,1], 'b.')
	subplot(2,2,3); axis('equal'); xlabel('X'); ylabel('Z')
	plot(-tmpXYZ3[:,0],tmpXYZ3[:,2], 'b.')

	# Whole body:
	figure(9)
	subplot(2,2,1); axis('equal'); xlabel('X'); ylabel('Y')
	plot(-xyz[:,1],xyz[:,0], 'b.')
	subplot(2,2,2); axis('equal'); xlabel('Z'); ylabel('Y')
	plot(-xyz[:,2],xyz[:,0], 'b.')
	subplot(2,2,3); axis('equal'); xlabel('X'); ylabel('Z')
	plot(-xyz[:,1],xyz[:,2], 'b.')



print "Time:", time.time()-time1





