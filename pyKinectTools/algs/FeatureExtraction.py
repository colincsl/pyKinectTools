
import os, time, sys
import numpy as np

from pyKinectTools.utils.DepthUtils import *


class Features:
	img = []
	labelImg = []
	personSlices = []
	labels = []
	allFeatureNames = ['basis', 'binary', 'viz', 'touchPoints']
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


	def extractClassificationFeatures(self, pData, featureLimits):
		# Recognition features
		com = pData['com']
		allCOMs = pData['data']['com']		
		center = np.array([-346.83551756, -16.10465379, 3475.0]) # new footage
		comRad = np.sqrt(np.sum((com - center)**2))
		comX = com[0]
		comY = com[1]
		comZ = com[2]

		frameCount = len(allCOMs)
		pTime = pData['elapsed']		

		if pTime < 5 or frameCount < 5:
			return -1		

		'''Get arclength'''
		arclength = 0
		for j in xrange(1, len(pData['data']['com'])):
			arclength += np.sqrt(np.sum((pData['data']['com'][j]-pData['data']['com'][j-1])**2))
		lengthTime = arclength / pTime

		if arclength < 1000:
			return -1

		''' --- Touch sensors --- '''
		''' Patient's head '''
		center = np.array([-146.83551756, -16.10465379, 3475.0])
		touchRad1 = np.min(np.sqrt(np.sum((allCOMs - center)**2, 1)))
		''' Patient's foot '''
		center = np.array([-141.13408991, -251.45427198, 2194.0])
		touchRad2 = np.min(np.sqrt(np.sum((allCOMs - center)**2, 1)))
		''' Ventilator '''
		center = np.array([-970.07707018, 57.13045534,  2754.])
		touchRad3 = np.min(np.sqrt(np.sum((allCOMs - center)**2, 1)))
		''' Computer '''
		center = np.array([-1402.64381383, -1025.05589005,  1570.0])
		touchRad4 = np.min(np.sqrt(np.sum((allCOMs - center)**2, 1)))

		'''Compare orientation vectors between objects'''
		ornFeatures = np.mean(pData['data']['ornCompare'], 0)

		''' Orientation Historgam '''
		allBasis = pData['data']['basis']
		ornHist = []
		h = allBasis
		# Seperate each axis from each frame
		h0 = np.array([x[0] for x in allBasis]) # X (+left)
		h1 = np.array([x[1] for x in allBasis]) # Y (+up)
		h2 = np.array([x[2] for x in allBasis]) # Z (+in)
		# Get planar rotations
		## About y axis (get x and z components because y is up)
		ang1 = np.array([np.arctan2(-x[1][0],x[1][2]) for x in h])*180.0/np.pi
		## About z axis
		ang2 = np.array([np.arctan2(-x[2][0],x[2][2]) for x in h])*180.0/np.pi
		a1 = np.minimum(ang1, ang2)
		a2 = np.maximum(ang1, ang2)
		# Ensure Y vector is pointing up
		validPoints = np.abs(h1[:,1])>.5  # Was previously set as h0
		ang1 = ang1[validPoints]; 
		a1 = a1[validPoints]
		ang1Hist, ang1HistInds = np.histogram(a1, 12, [-180, 180])
		ang1Hist = ang1Hist[:6]+ang1Hist[6:] #Collapse both directions
		ang1Hist = ang1Hist*1.0/np.max(ang1Hist)
		ornHist = ang1Hist

		''' Normalize '''
		arclength = (arclength - featureLimits['arclength'][0]) / (featureLimits['arclength'][1] - featureLimits['arclength'][0])
		lengthTime = (lengthTime - featureLimits['lengthTime'][0]) / (featureLimits['lengthTime'][1] - featureLimits['lengthTime'][0])
		touchRad1 = (touchRad1 - featureLimits['touchRad1'][0]) / (featureLimits['touchRad1'][1] - featureLimits['touchRad1'][0])
		touchRad2 = (touchRad2 - featureLimits['touchRad2'][0]) / (featureLimits['touchRad2'][1] - featureLimits['touchRad2'][0])
		touchRad3 = (touchRad3 - featureLimits['touchRad3'][0]) / (featureLimits['touchRad3'][1] - featureLimits['touchRad3'][0])
		touchRad4 = (touchRad4 - featureLimits['touchRad4'][0]) / (featureLimits['touchRad4'][1] - featureLimits['touchRad4'][0])
		frameCount = (frameCount - featureLimits['frameCount'][0]) / (featureLimits['frameCount'][1] - featureLimits['frameCount'][0])
		comRad = (comRad - featureLimits['comRad'][0]) / (featureLimits['comRad'][1] - featureLimits['comRad'][0])
		comX = (comX - featureLimits['comX'][0]) / (featureLimits['comX'][1] - featureLimits['comX'][0])
		comY = (comY - featureLimits['comY'][0]) / (featureLimits['comY'][1] - featureLimits['comY'][0])
		comZ = (comZ - featureLimits['comZ'][0]) / (featureLimits['comZ'][1] - featureLimits['comZ'][0])
		ornHist =  (ornHist - featureLimits['orn'][0]) / (featureLimits['orn'][1] - featureLimits['orn'][0])

		''' Compile into feature vector '''
		features = []
		features.append(arclength)
		features.append(lengthTime)
		features.append(touchRad1)
		features.append(touchRad2)
		features.append(touchRad3)
		features.append(touchRad4)
		features.append(comX)
		features.append(comY)
		features.append(comZ)
		for i in xrange(ornFeatures.shape[0]-3):
			features.append(ornFeatures[i])
		for i in xrange(ornHist.shape[0]):
			features.append(ornHist[i])
		
		return features






# def getExtrema(objects, labelInds, out, d, com, featureExt, ind):
# 	extrema = []
# 	mask = out[objects[ind]]==labelInds[ind]
# 	mask_erode = nd.binary_erosion(out[objects[ind]]==labelInds[ind], iterations=5)
# 	objTmp = np.array(d[objects[ind]])#, dtype=np.uint16)

# 	obj2Size = np.shape(objTmp)
# 	x = objects[ind][0].start # down
# 	y = objects[ind][1].start # right
# 	c = np.array([com[ind][0] - x, com[ind][1] - y])
# 	current = [c[0], c[1]]
# 	xyz = featureExt.xyz[ind]
	
# 	t = time.time()
# 	trailSets = []
# 	for i in xrange(15):
# 		if len(xyz) > 0:
# 			com_xyz = depth2world(np.array([[current[0]+x, current[1]+y, d[current[0]+x, current[1]+y]]]))[0]
# 			# pdb.set_trace()
# 			dists = np.sqrt(np.maximum(0, np.sum((xyz-com_xyz)**2, 1)))
# 			inds = featureExt.indices[ind]
			
# 			distsMat = np.zeros([obj2Size[0],obj2Size[1]], dtype=uint16)		
# 			distsMat = ((-mask)*499)
# 			distsMat[inds[0,:]-x, inds[1,:]-y] = dists 		
# 			objTmp = distsMat

# 			dists2 = np.empty([obj2Size[0]-2,obj2Size[1]-2,4], dtype=int16)
# 			dists2[:,:,0] = objTmp[1:-1, 1:-1] - objTmp[0:-2, 1:-1]#up
# 			dists2[:,:,1] = objTmp[1:-1, 1:-1] - objTmp[2:, 1:-1]#down
# 			dists2[:,:,2] = objTmp[1:-1, 1:-1] - objTmp[1:-1, 2:]#right
# 			dists2[:,:,3] = objTmp[1:-1, 1:-1] - objTmp[1:-1, 0:-2]#left
# 			dists2 = np.abs(dists2)


# 			dists2Tot = np.zeros([obj2Size[0],obj2Size[1]], dtype=int16)+9999		
# 			maxDists = np.max(dists2, 2)
# 			distThresh = 30
# 			outline = np.nonzero(maxDists>distThresh)
# 			mask[outline[0]+1, outline[1]+1] = 0

# 			dists2Tot[dists2Tot > 0] = 9999
# 			dists2Tot[-mask] = 15000
# 			dists2Tot[current[0], current[1]] = 0
# 			for t in trailSets:
# 				for i in t:
# 					dists2Tot[i[0], i[1]] = 0			

# 			visitMat = np.zeros_like(dists2Tot, dtype=uint8)
# 			visitMat[-mask] = 255

# 			trail = dijkstrasGraph.dijkstras(dists2Tot, visitMat, dists2, current)
# 			trailSets.append(trail)

# 			dists2Tot *= mask_erode

# 			maxInd = np.argmax(dists2Tot*(dists2Tot<500))
# 			maxInd = np.unravel_index(maxInd, dists2Tot.shape)

# 			extrema.append([maxInd[0]+x, maxInd[1]+y])
# 			current = [maxInd[0], maxInd[1]]
# 	# print (time.time()-t)/15

# 	# for i in extrema:
# 	# 	dists2Tot[i[0]-3:i[0]+3, i[1]-3:i[1]+3] = 499
# 	# imshow(dists2Tot*(dists2Tot < 500))
# 	# pdb.set_trace()

# 	return extrema


# Inter-person vector comparisons
# Always have n by 5 vector
def orientationComparison(vecs1, direc=2, size_=5):
	# Find the projection of each person's vector towards each other
	vecCompare = np.zeros([len(vecs1), size_])
	for i in xrange(len(vecs1)):
		for j in xrange(min(len(vecs1), size_)):
			if i == j:#j < i:
				continue
			else:
				vecCompare[i,j] = np.abs(np.dot(vecs1[i][direc], vecs1[j][direc]))
	vecCompare = -1.0*np.sort(-vecCompare, axis=1)
	return vecCompare




def calculateBasicPose(depthIm, mask):
	# Get nonzero indices and the corresponding depth values
	inds = np.nonzero(depthIm*mask > 0)
	depthVals = depthIm[inds]
	inds = np.vstack([inds, depthVals])

	xyz = depth2world(inds.T)
	com = xyz.mean(0)
	xyz -= com

	_, _, vh = np.linalg.svd(xyz, full_matrices=0)
	basis = vh.T
	if basis[1,0] < 0:
		basis = -1*basis

	return com, basis
