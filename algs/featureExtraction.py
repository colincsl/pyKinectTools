
import os, time, sys
import numpy as np
import cv, cv2
import scipy.ndimage as nd

from pyKinectTools.utils.icuReader import ICUReader
from pyKinectTools.algs.peopleTracker import Tracker
from pyKinectTools.utils.depthUtils import *
from pyKinectTools.algs.dijkstras import dijkstrasGraph


class Features:
	img = []
	labelImg = []
	personSlices = []
	labels = []
	allFeatureNames = ['basis', 'binary', 'viz']
	touchAreas = []
	segmentsTouched = []

	def __init__(self, featureList=[]):
		self.featureList = []
		self.addFeatures(featureList)

	def run(self, img, labelImg, personSlices, labels):
		self.img = img
		self.labelImg = labelImg
		self.personSlices = personSlices
		self.labels = labels
		self.segmentsTouched = []

		self.calculateFeatures()		
		return self.img, self.coms, self.bases, self.segmentsTouched


	def addFeatures(self, strList = []):
		for i in strList:
			if i in self.allFeatureNames:
				self.featureList.append(i)
			else:
				print i, " is not a valid feature name"

	def addTouchEvent(self, center, radius):
		if 'touch' not in self.featureList:
			self.featureList.append('touch')
		self.touchAreas.append([center, radius])

	# def addTouchBoxEvent(self, box):
	# 	self.featureList.append('touch')
	# 	self.touchAreas.append(box)		

	def calculateFeatures(self):
		if 'basis' in self.featureList:
			self.calculateBasis()
		if 'touch' in self.featureList:
			for i in self.touchAreas:
				self.touchEvent(i[0], i[1])
		if 'viz' in self.featureList:
			self.vizBasis()



	def calculateBasis(self):
		vecs_out = []
		com_out = [] #center of mass
		com_xyz_out = []
		bounds = []
		self.xyz = []
		self.indices = []
		if len(self.personSlices) > 0:
			for objIndex in xrange(len(self.personSlices)):
				inds = np.nonzero(self.labelImg[self.personSlices[objIndex]] == self.labels[objIndex])
				offsetX = self.personSlices[objIndex][0].start #Top left corner of object slice
				offsetY = self.personSlices[objIndex][1].start
				inds2 = [inds[0]+offsetX, inds[1]+offsetY]
				depVals = self.img[inds2]
				inds2.append(depVals)
				inds2 = np.transpose(inds2)
				# following is unnecessary unless axis has already been painted on the image
				# inds2 = [x for x in inds2 if x[2] != 0]
				xyz = np.array(depth2world(np.array(inds2)))
				inds2 = np.transpose(inds2)				
				self.xyz.append(xyz)
				self.indices.append(inds2)

				# Get bounding box
				x = [np.min(xyz[:,0]), np.max(xyz[:,0])]
				y = [np.min(xyz[:,1]), np.max(xyz[:,1])]
				z = [np.min(xyz[:,2]), np.max(xyz[:,2])]
				# bounds.append([[x[0],y[0],z[0]], [x[1],y[1],z[1]]])
				bounds.append([[x[0],y[0],z[0]], \
							   [x[0],y[0],z[1]], \
							   [x[0],y[1],z[0]], \
							   [x[0],y[1],z[1]], \
							   [x[1],y[0],z[0]], \
							   [x[1],y[0],z[1]], \
							   [x[1],y[1],z[0]], \
							   [x[1],y[1],z[1]]])

				com = xyz.mean(0)
				xyz = xyz - com

				u, s, v = np.linalg.svd(xyz, full_matrices=0)
				v = v.T
				if v[1,0] < 0:
					v = -1*v
				vecs = []
				for i in xrange(3):
					vecs.append(v[:,i])
				vecs_out.append(vecs)

				com_xyz_out.append(com)
				com_uv = world2depth(np.array([com]))
				com_uv = com_uv.T[0]
				com_out.append(com_uv)

		self.bases = vecs_out
		self.coms = com_out
		self.coms_xyz = com_xyz_out
		self.bounds = bounds


	def vizBasis(self):
		# Display axis on the original image
		for ind in range(len(self.coms)):
			com = self.coms_xyz[ind]
			v = np.array(self.bases[ind])
			spineStart = [com[0], com[1], com[2]]
			spineLine = []
			for axis in xrange(3):
				spine = v[:,axis]
				for i in xrange(-100*0, 100, 2):
					spineLine.append(spineStart+spine*i)
			spineLine = np.array(spineLine)
			spineLine_xyd = np.array(world2depth(spineLine))

			if 0:
				#Show 3d structure
				fig = figure(2)
				ax = fig.add_subplot(111, projection='3d')
				# ax.cla()
				ax.scatter(xyz[0,::4], xyz[1,::4], xyz[2,::4])
				ax.scatter(spineLine[:,0], spineLine[:,1], spineLine[:,2], 'g')
				xlabel('x')
				ylabel('y')
			
			if len(spineLine_xyd) > 0:
				# Make sure all points are within bounds
				inds = [x for x in range(len(spineLine_xyd[0])) if (spineLine_xyd[0, x] >= 0 and spineLine_xyd[0, x] < 480 and spineLine_xyd[1, x] >= 0 and spineLine_xyd[1, x] < 640)]
				spineLine_xyd = spineLine_xyd[:,inds]
				self.img[spineLine_xyd[0], spineLine_xyd[1]] = 0#spineLine_xyd[2]


	def touchEventBox(self, box=[]):
		# check if the person and event boxes intersect
		# box format: [[x1,y1,z1],[x2,y2,z1]] (two corners)
		segmentsTouched = []
		for ind in range(len(self.coms)):
			c = self.bounds[ind]
			# x
			# pdb.set_trace()
			if c[0][0] < box[0][0] < c[1][0] or c[0][0] < box[1][0] < c[1][0] \
				or box[0][0] < c[0][0] < box[1][0] or box[0][0] < c[1][0] < box[1][0]:

				# y
				if c[0][1] < box[0][1] < c[1][1] or c[0][1] < box[1][1] < c[1][1] \
					or box[0][1] < c[0][1] < box[1][1] or box[0][1] < c[1][1] < box[1][1]:
					# z
					if c[0][2] < box[0][2] < c[1][2] or c[0][2] < box[1][2] < c[1][2] \
						or box[0][2] < c[0][2] < box[1][2] or box[0][2] < c[1][2] < box[1][2]:
						segmentsTouched.append(ind)

		self.segmentsTouched.append(segmentsTouched)


	def touchEvent(self, center=[332, -104, 983], radius=50):
		# check if the person and event circle (w/in radius) overlap
		# pdb.set_trace()
		sqr_dist = np.sqrt(np.sum((self.bounds - np.array(center))**2, 2))
		touched = np.nonzero(np.any(sqr_dist < radius, 1))[0]

		self.segmentsTouched.append(touched)





	def play(self):
		# Look at histograms of centered data projected on each basis
		dists = np.sqrt(np.sum(xyz**2, 1))
		dists = np.sqrt((xyz**2))
		dx[inds2[0], inds2[1]] = dists[:, 0]
		# Project to each principal axis
		axis = 1
		d = np.asarray(np.asmatrix(dists) * np.asmatrix(v[:,axis]).T)
		dx = np.array(dx, dtype=np.int16)
		dx[inds2[0], inds2[1]] = -1*d[:,0]#-d[:, 0]
		figure(1)
		imshow(dx)
		h = np.histogram(d, 100, [-500, 500])
		figure(2)
		plot(h[1][1::],h[0])
		figure(3)
		plot(d)


def getExtrema(objects, labelInds, out, d, com, featureExt, ind):
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
	
# It crashes when the new person comes in

	t = time.time()
	trailSets = []
	for i in xrange(15):
		if len(xyz) > 0:
			com_xyz = depth2world(np.array([[current[0]+x, current[1]+y, d[current[0]+x, current[1]+y]]]))[0]
			# pdb.set_trace()
			dists = np.sqrt(np.maximum(0, np.sum((xyz-com_xyz)**2, 1)))
			inds = featureExt.indices[ind]
			
			distsMat = np.zeros([obj2Size[0],obj2Size[1]], dtype=uint16)		
			distsMat = ((-mask)*499)
			distsMat[inds[0,:]-x, inds[1,:]-y] = dists 		
			objTmp = distsMat

			dists2 = np.empty([obj2Size[0]-2,obj2Size[1]-2,4], dtype=int16)
			dists2[:,:,0] = objTmp[1:-1, 1:-1] - objTmp[0:-2, 1:-1]#up
			dists2[:,:,1] = objTmp[1:-1, 1:-1] - objTmp[2:, 1:-1]#down
			dists2[:,:,2] = objTmp[1:-1, 1:-1] - objTmp[1:-1, 2:]#right
			dists2[:,:,3] = objTmp[1:-1, 1:-1] - objTmp[1:-1, 0:-2]#left
			dists2 = np.abs(dists2)


			dists2Tot = np.zeros([obj2Size[0],obj2Size[1]], dtype=int16)+9999		
			maxDists = np.max(dists2, 2)
			distThresh = 30
			outline = np.nonzero(maxDists>distThresh)
			mask[outline[0]+1, outline[1]+1] = 0

			dists2Tot[dists2Tot > 0] = 9999
			dists2Tot[-mask] = 15000
			dists2Tot[current[0], current[1]] = 0
			for t in trailSets:
				for i in t:
					dists2Tot[i[0], i[1]] = 0			

			visitMat = np.zeros_like(dists2Tot, dtype=uint8)
			visitMat[-mask] = 255

			trail = dijkstrasGraph.dijkstras(dists2Tot, visitMat, dists2, current)
			trailSets.append(trail)

			dists2Tot *= mask_erode

			maxInd = np.argmax(dists2Tot*(dists2Tot<500))
			maxInd = np.unravel_index(maxInd, dists2Tot.shape)

			extrema.append([maxInd[0]+x, maxInd[1]+y])
			current = [maxInd[0], maxInd[1]]
	# print (time.time()-t)/15

	# for i in extrema:
	# 	dists2Tot[i[0]-3:i[0]+3, i[1]-3:i[1]+3] = 499
	# imshow(dists2Tot*(dists2Tot < 500))
	# pdb.set_trace()

	return extrema


# Inter-person vector comparisons
# Always have n by 5 vector
def orientationComparison(vecs1, direc=2, size_=5):
	# Find the projection of each person's vector towards each other
	# pdb.set_trace()
	vecCompare = np.zeros([len(vecs1), size_])
	for i in xrange(len(vecs1)):
		for j in xrange(min(len(vecs1), size_)):
			if i == j:#j < i:
				continue
			else:
				vecCompare[i,j] = np.abs(np.dot(vecs1[i][direc], vecs1[j][direc]))
	vecCompare = -1.0*np.sort(-vecCompare, axis=1)
	return vecCompare

