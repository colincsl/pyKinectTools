'''
Implements methods for background subtraction
--Adaptive Mixture of Gaussians
--Median Model
--other utility functions
'''
import os, time, sys
import numpy as np
import scipy.ndimage as nd
from IPython import embed
import cv2


def constrain(img, mini=-1, maxi=-1): #500, 4000
	if mini == -1:
		min_ = np.min(img[np.nonzero(img)])
	else:
		min_ = mini
	if maxi == -1:
		max_ = img.max()
	else:
		max_ = maxi

	img = np.clip(img, min_, max_)
	img -= min_
	if max_ == 0:
		max_ = 1
	img = np.array((img * (255.0 / (max_-min_))), dtype=np.uint8)

	img[img==img.max()] = 0

	return img

def extract_people_clusterMethod(img):
	from scipy.spatial import distance
	from sklearn.cluster import DBSCAN
	import cv2

	if len(img) == 1:
		img = img[0]
	img = img*nd.binary_opening(img>0, iterations=5)
	img = cv2.medianBlur(img, 5)
	# hist1 = np.histogram(img, 128)
	hist1 = np.histogram(img, 64)
	if (np.sum(hist1[0][1::]) > 100):
		samples = np.random.choice(np.array(hist1[1][1:-1], dtype=int), 1000, p=hist1[0][1::]*1.0/np.sum(hist1[0][1::]))
		samples = np.sort(samples)
	else:
		return np.zeros_like(img), []

	tmp = np.array([samples, np.zeros_like(samples)])
	D = distance.squareform(distance.pdist(tmp.T, 'cityblock'))
	# D = distance.squareform(distance.pdist(tmp.T, 'chebyshev'))
	S = 1 - (D / np.max(D))

	db = DBSCAN().fit(S, eps=0.95, min_samples=50)
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	clusterLimits = []
	# for i in range(-1, n_clusters_):
	max_ = 0
	for i in xrange(1, n_clusters_):		
		min_ = np.min(samples[np.nonzero((labels==i)*samples)])
		max_ = np.max(samples[np.nonzero((labels==i)*samples)])
		clusterLimits.append([min_,max_])
	if max_ != 255:
		clusterLimits.append([max_, 255])
	clusterLimits.append([0, min(clusterLimits)[0]])
	clusterLimits.sort()

	d = np.zeros_like(img)
	tmp = np.zeros_like(img)
	labels = []
	for i in xrange(0, n_clusters_):
		tmp = (img > clusterLimits[i][0])*(img < clusterLimits[i][1])
		d[np.nonzero((img > clusterLimits[i][0])*(img < clusterLimits[i][1]))] = (i+2)
		tmpLab = nd.label(tmp)
		if labels == []:
			labels = tmpLab
		else:
			labels = [labels[0]+((tmpLab[0]>0)*(tmpLab[0]+tmpLab[1])), labels[1]+tmpLab[1]]

	objs = nd.find_objects(labels[0])
	goodObjs = []
	for i in xrange(len(objs)):	
		if objs[i] != None:
			# px = nd.sum(d, labels[0], i)
			px = nd.sum(labels[0][objs[i]]>0)
			# print "Pix:", px
			if px > 7000:
				goodObjs.append([objs[i], i+1])

	d1A = np.zeros_like(img)
	for i in xrange(len(goodObjs)):
		## Plot all detected people
		# subplot(1, len(goodObjs)+1, i+1)
		# imshow(d[goodObjs[i][0]] == goodObjs[i][1])
		# imshow(d[goodObjs[i][0]] > 0)
		# imshow(labels[0] == goodObjs[i][1])

		d1A = np.maximum(d1A, (labels[0] == goodObjs[i][1])*(i+1))		
		# d1A[goodObjs[i][0]] = np.maximum(d1A[goodObjs[i][0]], (d[goodObjs[i][0]]>0)*8000)

	return d1A, goodObjs

def extract_people(im, mask, minPersonPixThresh=5000, gradThresh=None):
	'''
	---Paramaters---
	im : 
	mask : 
	minPersonPixThresh : 
	gradThresh : 
	gradientFilter : 

	---Returns---
	mask :
	userBoundingBoxes :
	userLabels :
	'''
	if gradThresh == None:
		grad_bin = mask
	else:
		gradients = np.gradient(im)
		mag = np.sqrt(gradients[0]**2+gradients[1]**2)
		mask *= mag<gradThresh

		# grad_g = np.max(np.abs(np.gradient(im.astype(np.int16))), 0)
		# grad_bin = (np.abs(grad_g) < gradThresh)
		# mask = mask*grad_bin# np.logical_and(mask[:,:-1],grad_bin)# np.logical_not(grad_bin))

	labelIm, maxLabel = nd.label(im*mask)
	connComps = nd.find_objects(labelIm, maxLabel)

	# Only extract if there are sufficient pixels and it is within a valid height/width ratio
	px_count = [nd.sum(labelIm[c]==l) for c,l in zip(connComps,range(1, maxLabel+1))]
	ratios = [(c[1].stop-c[1].start)/float(c[0].stop-c[0].start)for c in connComps]
	du = [float(c[0].stop-c[0].start) for c in connComps]
	dv = [float(c[1].stop-c[1].start) for c in connComps]
	areas = [float(c[0].stop-c[0].start)*float(c[1].stop-c[1].start)for c in connComps]
	# Filter
	usrTmp = [(c,l,px) for c,l,px,ratio,area in zip(connComps,range(1, maxLabel+1), px_count, ratios, areas) 
				if ratio < 2 and 
					px > minPersonPixThresh and
					px/area > 0.2
			]

	# if usrTmp != []:
		# print len(usrTmp), "users"
	# for i in range(len(px_count)):
		# if px_count[i] > minPersonPixThresh:
			# print px_count[i], np.array(px_count[i])/np.array(areas[i]), ratios[i]
	if len(usrTmp) > 0:
		userBoundingBoxes, userLabels, px_count = zip(*usrTmp)
	else:
		userBoundingBoxes = []
		userLabels = []
	userCount = len(userLabels)

	#Relabel foregound mask with multiple labels
	mask = im.astype(np.uint8)*0
	for i,i_new in zip(userLabels, range(1, userCount+1)):
		mask[labelIm==i] = i_new

	return mask, userBoundingBoxes, userLabels, px_count

def getMeanImage(depthImgs):
	mean_ = np.mean(depthImgs, 2)
	mean_ = mean_*(~nd.binary_dilation(mean_==0, iterations=3))

	## Close holes in images
	inds = nd.distance_transform_edt(mean_<500, return_distances=False, return_indices=True)
	i2 = np.nonzero(mean_<500)
	i3 = inds[:, i2[0], i2[1]]
	mean_[i2] = mean_[i3[0], i3[1]] # For all errors, set to avg 

	return mean_

def fillImage(im, tol=None):
	## Close holes in images
	inds = nd.distance_transform_edt(im==0, return_distances=False, return_indices=True)
	i2 = np.nonzero(im==0)
	i3 = inds[:, i2[0], i2[1]]
	im[i2] = im[i3[0], i3[1]]

	return im



def removeNoise(im, thresh=500):
	#Thresh is the envelope in the depth dimension within we remove noise
	zAvg = im[im[:,:,2]>0,2].mean()
	zThresh = thresh
	im[im[:,:,2]>zAvg+zThresh] = 0
	im[im[:,:,2]<zAvg-zThresh] = 0
	im[:,:,2] = nd.median_filter(im[:,:,2], 3)

	return im


''' Adaptive Mixture of Gaussians '''
class AdaptiveMixtureOfGaussians:

	def __init__(self, im, maxGaussians=5, learningRate=0.05, decayRate=0.25, variance=100**2):

		xRez, yRez = im.shape
		self.MaxGaussians = maxGaussians
		self.LearningRate = learningRate
		self.DecayRate = decayRate
		self.VarianceInit = variance
		self.CurrentGaussianCount = 1		

		self.Means = np.zeros([xRez,yRez,self.MaxGaussians])
		self.Variances = np.empty([xRez,yRez,self.MaxGaussians])
		self.Weights = np.empty([xRez,yRez,self.MaxGaussians])
		self.Deltas = np.empty([xRez,yRez,self.MaxGaussians])
		self.NumGaussians = np.ones([xRez,yRez], dtype=np.uint8)

		self.Deltas = np.zeros([xRez,yRez,self.MaxGaussians]) + np.inf

		self.Means[:,:,0] = im
		self.Weights[:,:,0] = self.LearningRate
		self.Variances[:,:,0] = self.VarianceInit

		self.Deviations = ((self.Means - im[:,:,np.newaxis])**2 / self.Variances)
		self.backgroundModel = im
		self.currentIm = im

	# @profile
	def update(self, im):

		self.currentIm = im
		self.currentIm[im == 0] = im.max()
		mask = im != 0		

		''' Check deviations '''
		self.Deviations = ((self.Means - im[:,:,np.newaxis])**2 / self.Variances) * mask[:,:,None]

		for m in range(self.CurrentGaussianCount):
			self.Deviations[m > self.NumGaussians,m] = np.inf

		Ownership = np.argmin(self.Deviations, -1)
		deviationMin = np.min(self.Deviations, -1)

		createNewMixture = deviationMin > 3
		createNewMixture[np.isinf(deviationMin)] = False
		replaceLowestMixture = np.logical_and(createNewMixture, self.NumGaussians>=self.MaxGaussians)
		createNewMixture = np.logical_and(createNewMixture, self.NumGaussians<self.MaxGaussians)

		''' Create new mixture using existing indices'''
		if np.any(createNewMixture):
			activeset_x, activeset_y = np.nonzero(createNewMixture)
			activeset_z = self.NumGaussians[activeset_x, activeset_y].ravel()

			# print "-------New Mixture------", len(activeset_x)
			self.Means[activeset_x,activeset_y,activeset_z] = im[activeset_x,activeset_y]
			self.Weights[activeset_x,activeset_y,activeset_z] = self.LearningRate
			self.Variances[activeset_x,activeset_y,activeset_z] = self.VarianceInit
			self.NumGaussians[activeset_x,activeset_y] += 1
			Ownership[activeset_x,activeset_y] = activeset_z

		''' Replace lowest weighted mixture '''
		if np.any(replaceLowestMixture):
			activeset_x, activeset_y = np.nonzero(replaceLowestMixture)

			activeset_z = np.argmin(self.Weights[activeset_x, activeset_y,:], -1)

			# print "-------Replace Mixture------", len(activeset_x)
			self.Means[activeset_x,activeset_y,activeset_z] = im[activeset_x,activeset_y]
			self.Weights[activeset_x,activeset_y,activeset_z] = self.LearningRate
			self.Variances[activeset_x,activeset_y,activeset_z] = self.VarianceInit
			Ownership[activeset_x,activeset_y] = activeset_z

		self.CurrentGaussianCount = self.NumGaussians.max()
		# print "Gaussians: ", self.NumGaussians.max()


		''' Update gaussians'''
		for m in range(self.CurrentGaussianCount):
			self.Deltas[:,:,m]		= im - self.Means[:,:,m]
			tmpOwn 					= Ownership==m
			# print "Own:",np.sum(tmpOwn)

			self.Weights[:,:,m]		= self.Weights[:,:,m] 	+ self.LearningRate*(tmpOwn - self.Weights[:,:,m]) - self.LearningRate*self.DecayRate			
			tmpWeight 				= tmpOwn*(self.LearningRate/self.Weights[:,:,m])			
			tmpMask = (self.Weights[:,:,m]<=0.001)
			tmpWeight[tmpMask] = 0

			self.Means[:,:,m] 		= self.Means[:,:,m] 	+ tmpWeight * self.Deltas[:,:,m]
			# self.Variances[:,:,m] 	= self.Variances[:,:,m] + tmpWeight * (self.Deltas[:,:,m]**2 - self.Variances[:,:,m])

			''' If the weight is zero, reset '''
			# embed()
			if m < np.any(tmpMask):
				if self.CurrentGaussianCount > 1:
					activeset_x, activeset_y = np.nonzero(tmpMask * (self.NumGaussians > m))
					try:
						# self.Variances[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = self.Variances[activeset_x, activeset_y, m+1:self.CurrentGaussianCount]
						self.Means[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = self.Means[activeset_x, activeset_y, m+1:self.CurrentGaussianCount]
						self.Weights[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = self.Means[activeset_x, activeset_y, m+1:self.CurrentGaussianCount]
						# self.Variances[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = 0
						self.Means[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = 0
						self.Weights[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = 0
						self.NumGaussians[activeset_x, activeset_y] -= 1
						# print "Reduce gaussians on slice", m, "max:", self.NumGaussians.max()
					except:
						embed()



		# print np.sum(self.Weights[:,:,0]< .1)
		# embed()
		self.backgroundModel = np.max(self.Means, 2)
		# self.backgroundModel = np.nanmax(self.Means, 2)
		
		'''This'''
		# tmp = np.argmax(self.Weights,2).ravel()
		# self.backgroundModel = self.Means[:,:,tmp]

		# self.backgroundModel = self.Means[:,:,np.nanargmax(self.Weights, 2).ravel()]
		# self.backgroundModel = np.nanmax(self.Means*(self.Means<10000), 2)


	def getModel(self):
		return self.backgroundModel

	def getForeground(self, thresh=100):
		# mask = self.currentIm!=0
		residual = np.abs(self.currentIm - self.backgroundModel)
		# residual = self.backgroundModel
		# residual *= mask
		foreground = residual > thresh
		foreground = nd.binary_closing(foreground, iterations=1)
		# import cv2
		# cv2.imshow("res", residual/residual.max())
		return foreground


class MedianModel:

	def __init__(self, depthIm):
		self.prevDepthIms = fillImage(depthIm.copy())[:,:,None]
		self.backgroundModel = self.prevDepthIms[:,:,0]

	def update(self,depthIm):

		# Add to set of images
		self.currentIm = fillImage(depthIm.copy())
		self.prevDepthIms = np.dstack([self.prevDepthIms, self.currentIm])

		# Check if too many (or few) images
		imCount = self.prevDepthIms.shape[2]
		# if imCount <= 1:
			# return
		if imCount > 50:
			self.prevDepthIms = self.prevDepthIms[:,:,-50:]

		self.backgroundModel = np.median(self.prevDepthIms, -1)

	def getModel(self):
		return self.backgroundModel		

	def getForeground(self, thresh=100):
		return (self.backgroundModel - self.currentIm) > thresh



''' Likelihood based on optical flow'''

# backgroundProb[foregroundMask == 0] = 1.
# prevDepthIms[:,:,-1] = 5000.

# if backgroundProbabilities is None:
# 	backgroundProbabilities = backgroundProb[:,:,np.newaxis]
# else:
# 	backgroundProbabilities = np.dstack([backgroundProbabilities, backgroundProb])

# cv2.imshow("flow_color", flow[:,:,0]/float(flow[:,:,0].max()))
# cv2.imshow("Prob", backgroundProbabilities.mean(-1) / backgroundProbabilities.mean(-1).max())
# tmp = np.sum(backgroundProbabilities*prevDepthIms[:,:,1:],2) / np.sum(backgroundProbabilities,2)
# # print tmp.min(), tmp.max()
# cv2.imshow("BG_Prob", tmp/tmp.max())

# backgroundProbabilities -= .01
# backgroundProbabilities = np.maximum(backgroundProbabilities, 0)

# prob = .5
# # backgroundModel = tmp#(tmp > prob)*tmp + (tmp < prob)*5000
# backgroundModel = np.median(prevDepthIms, 2)
# cv2.imshow("BG model", backgroundModel/5000.)




# backgroundProb = np.exp(-flowMag)











# else:
# 	mask = np.abs(backgroundModel.astype(np.int16) - depthIm) < 50
# 	mask[depthIm < 500] = 0

# 	depthBG = depthIm.copy()
# 	depthBG[~mask] = 0
# 	if backgroundTemplates.shape[2] < backgroundCount or np.random.rand() < bgPercentage:
# 		# mask = np.abs(backgroundTemplates[0].astype(np.int16) - depthIm) < 20
# 		backgroundTemplates = np.dstack([backgroundTemplates, depthBG])
# 		backgroundModel = backgroundTemplates.sum(-1) / np.maximum((backgroundTemplates>0).sum(-1), 1)
# 		backgroundModel = nd.maximum_filter(backgroundModel, np.ones(2))
# 	if backgroundTemplates.shape[2] > backgroundCount:
# 		# backgroundTemplates.pop(0)
# 		backgroundTemplates = backgroundTemplates[:,:,1:]

# 	depthIm[mask] = 0


''' Background model #2 '''
# mask = None
# if backgroundModel is None:
# 	backgroundModel = depthIm.copy()
# 	backgroundTemplates = depthIm[:,:,np.newaxis].copy()
# else:
# 	mask = np.abs(backgroundModel.astype(np.int16) - depthIm)
# 	mask[depthIm < 500] = 0

# 	depthBG = depthIm.copy()
# 	# depthBG[~mask] = 0
# 	if backgroundTemplates.shape[2] < backgroundCount or np.random.rand() < bgPercentage:
# 		# mask = np.abs(backgroundTemplates[:,:,0].astype(np.int16) - depthIm)# < 20
# 		backgroundTemplates = np.dstack([backgroundTemplates, depthBG])
# 		backgroundModel = backgroundTemplates.sum(-1) / np.maximum((backgroundTemplates>0).sum(-1), 1)
# 		# backgroundModel = nd.maximum_filter(backgroundModel, np.ones(2))
# 	if backgroundTemplates.shape[2] > backgroundCount:
# 		backgroundTemplates = backgroundTemplates[:,:,1:]

	# depthIm[mask] = 0
