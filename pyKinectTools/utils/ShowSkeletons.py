import os
import optparse
import cPickle as pickle
import cv2
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd
import skimage
from skimage import feature
from skimage import color
from pyKinectTools.utils.DepthUtils import world2depth, depthIm2XYZ
from pyKinectTools.algs.HistogramOfOpticalFlow import getFlow, hof
from pyKinectTools.algs.BackgroundSubtraction import AdaptiveMixtureOfGaussians, fillImage, extractPeople
from pyKinectTools.algs.FeatureExtraction import calculateBasicPose
# from pyKinectTools.utils.RealtimeReader import *

from mayavi import mlab
figure = mlab.figure(1, bgcolor=(0,0,0), fgcolor=(1,1,1))
# figure = mlab.figure(1, bgcolor=(1,1,1), fgcolor=(0,0,0))
# figure = mlab.gcf()
mlab.clf()
figure.scene.disable_render = True

from IPython import embed
import cProfile

from time import time
timeStart = time()



np.seterr(divide='ignore')


# ''' Find temporal sub-second offset '''
Cam_OFFSETS = [0, 18, 13]


def saveFeatures(featureDict):

	featureList = featureDict.items()
	labels = [x[0] for x in featureList[0]]

	data = []
	for i in range(len(featureList)):
		data.append([x[1] for x in featureList[i]])

	np.savez("/media/Data/allFeatures_tmp", labels=labels, data=data)

def loadFeatures(filename):
	file_ = np.load(filename)
	labels = file_['labels']
	data = file_['data']

	return labels, data

# from sklearn.decomposition import FastICA
def learnICADict(features, components=25):
	from sklearn.decomposition import FastICA

	icaHOG = FastICA(n_components=components)
	icaHOF = FastICA(n_components=components)

	icaHOG.fit(np.array([x['hog'] for x in features]).T)
	icaHOF.fit(np.array([x['hof'] for x in features]).T)

	hogComponents = icaHOG.components_.T
	hofComponents = icaHOF.components_.T

	return hogComponents, hofComponents

# from sklearn.decomposition import FastICA
def learnNMFDict(features, components=25):
	from sklearn.decomposition import NMF

	nmfHOG = NMF(n_components=components)
	nmfHOF = NMF(n_components=components)

	nmfHOG.fit(np.array([x['hog'] for x in features]).T)
	nmfHOF.fit(np.array([x['hof'] for x in features]).T)

	hogComponents = icaHOG.components_.T
	hofComponents = icaHOF.components_.T

	return hogComponents, hofComponents	

def displayComponents(components):
	sides = ceil(np.sqrt(len(components)))
	for i in range(len(components)):
		subplot(sides, sides, i+1)
		imshow(hog2image(components[i]))



def computeFeaturesWithSkels(image, users=None, flow=None, device=2, computeHog=True, computeLBP=False, vis=False):

	features = {}

	# Head, Left hand, right hand
	bodyIndicies = [0, 5, 8] 

	for u in users.keys():
		xyz = users[u]['com']
		uvw = world2depth(xyz)
		hogs = []

		# Only plot if there are valid coordinates (not [0,0,0])
		if uvw[0] > 0 and users[u]['tracked'] and len(users[u]['jointPositions'].keys()) > 0:

			''' HOG at each Body positions '''
			for i in bodyIndicies:
				pt = world2depth(users[u]['jointPositions'][users[u]['jointPositions'].keys()[i]])
				uImg = image[pt[0]/2-textureSize[0]/2:pt[0]/2+textureSize[0]/2, (pt[1]/2-textureSize[1]/2):(pt[1]/2+textureSize[1]/2)]

				if uImg.size == 0 or uImg.max()==0:
					return

				# Ensure it's the right size
				uImg = np.ascontiguousarray(uImg)
				uImg.resize(textureSize)

				# Scale and prevent blooming
				uImg = np.copy(uImg).astype(np.float)
				uImg[uImg>0] -= np.min(uImg[uImg>0])
				# This reduces problems with stark changes in background
				uImg = np.minimum(uImg, 100)
				# uImg /= (np.max(uImg)/255.)

				fillImage(uImg)
				if not vis:
					hogArray = feature.hog(uImg, visualise=False)
				else:
					hogArray, hogIm = feature.hog(uImg, visualise=True)
					cv2.namedWindow("hog_"+str(i))
					cv2.imshow("hog_"+str(i), hogIm)
					# cv2.imshow("hog_"+str(i), uImg)
					ret = cv2.waitKey(10)

				hogs.append(hogArray)

		if uvw[0] > 0:
			features[u] = {
							'com':xyz,
							'time':users[u]['timestamp'],
							'device':device,
							'hogs':hogs
							}

	return features

def computeUserFeatures(colorIm, depthIm, flow, boundingBox, mask=None, windowSize=[96,72], visualise=False):
	''' 
		images should be bounding box of person
		mask should be of person within bounding box
	'''

	assert colorIm.shape[0:2] == depthIm.shape and flow.shape[0:2] == depthIm.shape, "Wrong dimensions when computing features"

	''' Extract User from images '''
	if mask is not None:
		colorUserIm = np.ascontiguousarray(colorIm[boundingBox]*mask[boundingBox][:,:,np.newaxis])
		depthUserIm = np.ascontiguousarray(depthIm[boundingBox]*mask[boundingBox])
		flowUserIm = np.ascontiguousarray(flow[boundingBox]*mask[boundingBox][:,:,np.newaxis])

		# com = np.array(nd.center_of_mass(mask, 1), dtype=np.int)
		# com = depth2world(np.array([[com[0], com[1], depthIm[com[0], com[1]]]]))
	else:
		colorUserIm = np.ascontiguousarray(colorIm[boundingBox])
		depthUserIm = np.ascontiguousarray(depthIm[boundingBox])
		flowUserIm = np.ascontiguousarray(flow[boundingBox])

		x = (boundingBox[0].stop+boundingBox[0].start)/2
		y = (boundingBox[1].stop+boundingBox[1].start)/2
		# com = depth2world(np.array([[x,y, depthIm[x,y]]]))
	''' Resize images '''
	colorUserIm = sm.imresize(colorUserIm, [windowSize[0],windowSize[1],3])
	depthUserIm = sm.imresize(depthUserIm, windowSize)	
	flowUserImTmp0 = sm.imresize(flowUserIm[:,:,0], windowSize)
	flowUserImTmp1 = sm.imresize(flowUserIm[:,:,1], windowSize)
	flowUserIm = np.dstack([flowUserImTmp0,flowUserImTmp0])

	''' Get User Center of Mass and Orientation '''
	com, ornBasis = calculateBasicPose(depthIm, mask)

	''' Get HOG, HOF, color histogram '''
	colorIm_g = colorUserIm.mean(-1)

	if visualise:
		colorHistograms = [np.histogram(colorUserIm[:,:,i], bins=20, range=(0,255))[0] for i in range(3)]
		hogArray, hogIm = feature.hog(colorIm_g, visualise=True, orientations=4)
		hofArray, hofIm = hof(flowUserIm, visualise=True, orientations=5)

		features = {'com':com,
					'ornBasis':ornBasis,
					'hog':hogArray, 'hogIm':hogIm,
					'hof':hofArray, 'hofIm':hofIm,
					'colorHistograms':colorHistograms
					}
	else:
		hogArray, hogIm = feature.hog(colorIm_g, orientations=4)
		hofArray, hofIm = hof(flowUserIm, orientations=5)

		features = {'com':com, 
					'ornBasis':ornBasis,
					'hog':hogArray,
					'hof':hofArray,
					'colorHistograms':colorHistograms
					}

	return  features



def plotUsers(image, users=None, flow=None, vis=True, device=2, backgroundModel=None, computeHog=True, computeLBP=False):

	usersPlotted = 0
	uvw = [-1]

	bodyIndicies = [0, 5, 8] # See SkeletonUtils.py
	hogs = []
	for u in users.keys():
		if users[u]['tracked']:
			xyz = users[u]['com']
			uvw = world2depth(xyz)

			''' Only plot if there are valid coordinates (not [0,0,0])'''
			if uvw[0] > 0:
				if users[u]['tracked'] and len(users[u]['jointPositions'].keys()) > 0:

					'''Colorize COM'''
					cv2.rectangle(image, tuple([uvw[1]/2-3, uvw[0]/2-3]), tuple([uvw[1]/2+3, uvw[0]/2+3]), (4000))

					'''Plot skeleton'''
					if 1:
						w = 3
						for j in users[u]['jointPositions'].keys():
							pt = world2depth(users[u]['jointPositions'][j])
							image[pt[0]/2-w:pt[0]/2+w, pt[1]/2-w:pt[1]/2+w] = 4000                                                        

					usersPlotted += 1

	if vis and usersPlotted >= 0:
		# Make sure windows open
		cv2.namedWindow('Depth_'+str(device))
		cv2.imshow('Depth_'+str(device), image.astype(np.float)/5000.0)
		ret = cv2.waitKey(10)

	return ret


def format(x):
	if len(x) == 1:
		return x+'0'
	else:
		return x


class multiCameraTimeline:

	def __init__(self, files):
		self.files = files
		self.stamp = np.zeros(len(files), dtype=int)
		self.fileLengths = [len(x) for x in files]
		self.devCount = len(files)

	def __iter__(self):

		while 1:
			# Get latest file names/times
			tmpFiles = [self.files[i][self.stamp[i]] if self.stamp[i]<self.fileLengths[i] else np.inf for i in xrange(self.devCount)]
			# sec = [int(x.split('_')[-2])*100 if type(x)==str else np.inf for x in tmpFiles]
			# ms = [int(format(x.split('_')[-1][:x.split('_')[-1].find('.')])) if type(x)==str else np.inf for x in tmpFiles]
			# ms = [(x+MS_OFFSET)%100 for x in ms]
			sec = [int(x.split('_')[-4])*100 if type(x)==str else np.inf for x in tmpFiles]
			ms = [int(x.split('_')[-3]) if type(x)==str else np.inf for x in tmpFiles]			
			times = [s*100 + m for s,m in zip(sec, ms)]
			# times = [int(x.split('_')[-2])*100 + int(format(x.split('_')[-1][:x.split('_')[-1].find('.')])) if type(x)==str else np.inf for x in tmpFiles]

			# Account for time difference between the cameras
			for i in xrange(self.devCount):
				times[i] += Cam_OFFSETS[i]*100 

			#Find the earliest frame
			dev = np.argsort(times)[0]
			#If there are no more frames 
			if self.stamp[dev] >= self.fileLengths[dev]:
				dev = None
				for d, i in zip(np.argsort(times), range(len(times))):
					if d < self.fileLengths[i]:
						dev = d
						break
				if dev == None:
					raise StopIteration()

			self.stamp[dev] += 1
			# If at the end of the roll
			if tmpFiles[dev] == np.inf:
				raise StopIteration()
			# if dev == 2:
			# 	print times[dev], tmpFiles[dev], self.stamp, dev
			yield dev, tmpFiles[dev]



# -------------------------MAIN------------------------------------------
# @profile
def main(getDepth, getColor, getSkel, getMask, calculateFeatures):

	# getDepth = True
	# getColor = False
	# getSkel = False
	# getMask = False
	# calculateFeatures = False

	ret = 0
	backgroundTemplates = np.empty([1,1,1])
	backgroundModel = None
	backgroundCount = 20
	bgPercentage = .05
	prevDepthIm = None
	prevDepthIms = []
	prevColorIms = []

	dayDirs = os.listdir('depth/')
	dayDirs = [x for x in dayDirs if x[0]!='.']
	dayDirs.sort(key=lambda x: int(x))

	allFeatures = []

	for dayDir in dayDirs:
		hourDirs = os.listdir('depth/'+dayDir)
		hourDirs = [x for x in hourDirs if x[0]!='.']
		hourDirs.sort(key=lambda x: int(x))


		for hourDir in hourDirs: # Hours
			minuteDirs = os.listdir('depth/'+dayDir+'/'+hourDir)
			minuteDirs = [x for x in minuteDirs if x[0]!='.']
			minuteDirs.sort(key=lambda x: int(x))
			for minuteDir in minuteDirs: # Minutes

				if minuteDir[0] == '.': # Prevent from reading hidden files
					continue

				depthFiles = []
				skelFiles = []

				# For each available device:
				devices = os.listdir('depth/'+dayDir+'/'+hourDir+'/'+minuteDir)
				devices = [x for x in devices if x[0]!='.' and x.find('tmp')<0]
				devices.sort()
				for deviceID in ['device_1']:#devices:
					# Get filenames # Seconds
					''' Sort files '''					
					if getDepth:
						depthTmp = os.listdir('depth/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+deviceID)
						tmpSort = [int(x.split('_')[-3])*100 + int(format(x.split('_')[-2])) for x in depthTmp]			
						depthTmp = np.array(depthTmp)[np.argsort(tmpSort)].tolist()
						depthFiles.append([x for x in depthTmp if x.find('.png')>=0])
					if getSkel:
						skelTmp = os.listdir('skel/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+deviceID)			
						tmpSort = [int(x.split('_')[-4])*100 + int(format(x.split('_')[-3])) for x in skelTmp]			
						skelTmp = np.array(skelTmp)[np.argsort(tmpSort)].tolist()			
						skelFiles.append([x for x in skelTmp if x.find('.dat')>=0])



				for dev, depthFile in multiCameraTimeline(depthFiles):
					# try:
					if 1:
						print depthFile

						# if backgroundTemplates.shape[2] == backgroundCount and len(coms) <= 1:

						''' Load Depth '''
						if getDepth:
							depthIm = sm.imread('depth/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'+depthFile)
							depthIm = np.array(depthIm, dtype=np.uint16)
						''' Load Color '''
						if getColor:
							colorFile = 'color_'+depthFile[6:-4]+'.jpg'
							colorIm = sm.imread('color/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'+colorFile)
							# colorIm_g = colorIm.mean(-1, dtype=np.uint8)
							colorIm_g = skimage.img_as_ubyte(skimage.color.rgb2gray(colorIm))
							# colorIm_lab = skimage.color.rgb2lab(colorIm).astype(np.uint8)
						''' Load Mask '''
						# if getMask:
						# 	maskIm = sm.imread('depth/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'+depthFile[:-4]+"_mask.jpg") > 100
						# 	depthIm = depthIm*(1-maskIm)+maskIm*5000

						''' Load Skeleton Data '''
						if getSkel:
							skelFile = 'skel_'+depthFile[6:-4]+'_.dat'
							if os.path.isfile('skel/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'+skelFile):
								with open('skel/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'+skelFile, 'rb') as inFile:
									users = pickle.load(inFile)				
							else:
								print "No user file:", skelFile

							coms = [users[x]['com'] for x in users.keys() if users[x]['com'][2] > 0.0]

							jointCount = 0
							for i in users.keys():
								user = users[i]
								# if user['jointPositions'][user['jointPositions'].keys()[0]] != -1:
									# print user['jointPositionsConfidence']
									# jointCount = 1

						depthIm = np.minimum(depthIm.astype(np.float), 5000)
						fillImage(depthIm)


						'''Background model'''
						if backgroundModel is None:
							bgSubtraction = AdaptiveMixtureOfGaussians(depthIm, maxGaussians=3, learningRate=0.01, decayRate=0.02, variance=300**2)
							backgroundModel = bgSubtraction.getModel()
							if getColor:
								prevColorIm = colorIm_g.copy()
							continue
						else:
							bgSubtraction.update(depthIm)

						backgroundModel = bgSubtraction.getModel()
						foregroundMask = bgSubtraction.getForeground(thresh=50)

						''' Find people '''
						if getSkel:
							ret = plotUsers(depthIm, users, device=devices[dev], vis=True)
						if getMask:
							foregroundMask, userBoundingBoxes, userLabels = extractPeople(depthIm, foregroundMask, minPersonPixThresh=1000, gradientFilter=True, gradThresh=75)
						
						''' Calculate user features '''
						if calculateFeatures:
							''' Color Optical Flow '''
							flow = getFlow(prevColorIm, colorIm_g)
							prevColorIm = colorIm_g.copy()
							
							userCount = len(userBoundingBoxes)
							for i in xrange(userCount):
								userBox = userBoundingBoxes[i]
								userMask = foregroundMask==i+1
								allFeatures.append(computeUserFeatures(colorIm, depthIm, flow, userBox, mask=userMask, windowSize=[96,72], visualise=True))
						''' Or get CoM + orientation '''
						if getMask:
							coms = []
							orns = []
							userCount = len(userBoundingBoxes)
							for i in xrange(userCount):
								userBox = userBoundingBoxes[i]
								userMask = foregroundMask==i+1
								com, ornBasis = calculateBasicPose(depthIm, userMask)
								coms.append(com)
								orns.append(ornBasis[1])


						if getDepth:
							cv2.putText(depthIm, "Day "+dayDir+" Time "+hourDir+":"+minuteDir+" Dev#"+str(dev), (10,220), cv2.FONT_HERSHEY_DUPLEX, 0.6, 5000)					
							cv2.imshow("Depth", depthIm/5000.)
						if getColor:
							# cv2.putText(colorIm, "Day "+dayDir+" Time "+hourDir+":"+minuteDir+" Dev#"+str(dev), (10,220), cv2.FONT_HERSHEY_DUPLEX, 0.6, 5000)					
							cv2.imshow("I_orig", colorIm)
							if calculateFeatures:
								cv2.imshow("I", colorIm*foregroundMask[:,:,np.newaxis])
								cv2.imshow("I_masked", colorIm + (255-colorIm)*(((foregroundMask)[:,:,np.newaxis])))
						if getMask:
							cv2.imshow("Mask", foregroundMask.astype(np.float)/float(foregroundMask.max()))
							cv2.imshow("BG Model", backgroundModel.astype(np.float)/float(backgroundModel.max()))

						''' Top Down View '''
						if 0 and getMask:
							from pyKinectTools.algs.Normals import getTopdownMap
							topDownView = getTopdownMap(depthIm, rez=500)
							cv2.imshow("Top Down", topDownView)

						if 1:
							#3D Vis
							# figure = mlab.figure(1, fgcolor=(1,1,1), bgcolor=(0,0,0))
							# from pyKinectTools.utils.DepthUtils import *
							pts = depthIm2XYZ(depthIm).astype(np.int)
							interval = 25
							figure.scene.disable_render = True
							mlab.clf()
							# ss = mlab.points3d(-pts[::interval,0], pts[::interval,1], pts[::interval,2], colormap='Blues', vmin=1000., vmax=5000., mode='2dvertex')
							ss = mlab.points3d(pts[::interval,0], pts[::interval,1], pts[::interval,2], 5.-(np.minimum(pts[::interval,2], 5000)/float((-pts[:,2]).max()))/1000., scale_factor=25., colormap='Blues')#, mode='2dvertex')
							# , scale_factor=25.
							mlab.view(azimuth=0, elevation=0, distance=3000., focalpoint=(0,0,0), figure=figure)#, reset_roll=False)
							# mlab.roll(90)
							currentView = mlab.view()
							figure.scene.disable_render = False
							mlab.draw()
							# mlab.show()
							# ss = mlab.points3d(pts[::interval,0], pts[::interval,1], pts[::interval,2], color=col, scale_factor=5)
							# ss = mlab.points3d(pts[:,0], pts[:,1], pts[:,2], color=(1,1,1), scale_factor=5)

							# from IPython import embed
							# embed()

							# ss = mlab.points3d(pts[:,0], pts[:,1], pts[:,2])

						ret = cv2.waitKey(10)

						# prevDepthIms.append(depthIm.copy())
						# prevColorIms.append(colorIm_g)

						if ret > 0:
							break
					# except:
					# 	print "Erroneous frame"
					# 	cv2.imshow("D", depthIm.astype(np.float)/5000)
					# 	ret = cv2.waitKey(10)

				if ret > 0:
					break
			if ret > 0:
				break
		if ret > 0:
			# embed()
			break

	embed()


if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-s', '--skel', dest='skel', action="store_true", default=False, help='Enable skeleton')	
	parser.add_option('-d', '--depth', dest='depth', action="store_true", default=False, help='Enable depth images')		
	parser.add_option('-c', '--color', dest='color', action="store_true", default=False, help='Enable color images')	
	parser.add_option('-m', '--mask', dest='mask', action="store_true", default=False, help='Enable enternal mask')			
	parser.add_option('-f', '--calcFeatures', dest='bgSubtraction', action="store_true", default=False, help='Enable feature extraction')				
	(opt, args) = parser.parse_args()

	if len(args) > 0:
		print "Wrong input argument"
	else:
		main(getDepth=opt.depth, getSkel=opt.skel, getColor=opt.color, getMask=opt.mask, calculateFeatures=opt.bgSubtraction)

	'''Profiling'''
	# cProfile.runctx('main()', globals(), locals(), filename="ShowSkeletons.profile")




''' Compare non-filled with filled image re: hog and lbp '''
# figure(1); imshow(depthIm)
# hArray,hIm = feature.hog(depthIm, visualise=True)
# lIm = feature.local_binary_pattern(depthIm, 20, 5, 'uniform')
# figure(3); imshow(hIm)
# figure(5); imshow(lIm)

# depthIm = fillImage(depthIm)

# # figure(2); imshow(depthIm)
# hArray,hIm = feature.hog(depthIm, visualise=True)
# lIm = feature.local_binary_pattern(depthIm, 20, 5, 'uniform')
# depthIm = hIm
# figure(4); imshow(hIm)
# figure(6); imshow(lIm)

# seg.quickshift(np.repeat((depthIm[:,:,newaxis]/200).astype(np.uint8), 3, 2))



# np.save("../../AllFeatures", allFeatures_perUser)