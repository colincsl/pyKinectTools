from copy import deepcopy
import sys, os, time
import cv2, numpy as np
# sys.path.append('/Users/colin/libs/visionTools/slic-python/')
# import slic
# import Image 
# from pyKinectTools.algs.skeletonBeliefPropagation import *
from pyKinectTools.algs.graphAlgs import *
from pyKinectTools.algs.geodesicSkeleton import *
import pyKinectTools.algs.neighborSuperpixels.neighborSuperpixels as nsp


def roughOrientationEstimate(xyz):
	U,_,vT=np.linalg.svd((xyz-xyz.mean(0)), full_matrices=0)
	forwardVector = vT[2,:]

	return forwardVector, vT





def orientationEstimate(im, regions, regionXYZ, regionPos):

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





