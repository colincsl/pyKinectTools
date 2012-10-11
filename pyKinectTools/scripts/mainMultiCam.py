

import os, time, sys, cPickle
import numpy as np
import cv, cv2
# People extraction/tracking
from pyKinectTools.algs.BackgroundSubtraction import *
from pyKinectTools.algs.PeopleTracker import Tracker
# from pyKinectTools.algs.PersonTracker import Tracker
from pyKinectTools.algs.FeatureExtraction import *
from pyKinectTools.algs.GlobalSignalSystem import *
# Skeletal
# from pyKinectTools.algs.GeodesicSkeleton import *
# from pyKinectTools.algs.PictorialStructures import *
from pyKinectTools.algs.Manifolds import *
# Utils
from pyKinectTools.algs.Normals import *
from pyKinectTools.utils.RealtimeReader import *
from pyKinectTools.utils.DepthUtils import *
from pyKinectTools.utils.DepthReader import DepthReader
# Classifiers
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm as SVM
# Voice
from Foundation import *
import AppKit

'''------------ Setup Kinect ------------'''
depthConstraint = [500, 5000]
if 0:
	''' Physical Kinect '''
	depthDevice = RealTimeDevice()
	depthDevice.addDepth(depthConstraint)
	depthDevice.addColor()
	depthDevice.start()
	depthDevice.setMaxDist(depthConstraint[1])
	depthDevice.generateBackgroundModel()
else:
	''' ICU Data '''	
	path = '/Users/colin/data/ICU_7May2012_Wide/'
	startTime = 8000 #wide, 	
	# path = '/Users/colin/data/ICU_7May2012_Close/'
	# startTime = 1600 #close
	framerate = 30
	depthDevice = DepthReader(path, framerate, startTime, constrain=depthConstraint)

	''' Get background model '''
	depthImgs1 = []
	depthStackIndepthRaw = 0
	for i in xrange(10):
		depthDevice.update()
		depthImgs1.append(depthDevice.depthDataRaw)

	depthImgs1 = np.dstack(depthImgs1)
	meanIm = getMeanImage(depthImgs1)
	# m1 = constrain(mean1, 500, 2000)
	meanIm8 = constrain(meanIm, depthConstraint[0], depthConstraint[1]) # [500,2000] for closeup
	meanIm8[meanIm8==meanIm8.max()] = 0

	startTime = 7200#6900  #OLD:#2000 #2x procedure: 6900, 7200
	# startTime = 600#6900  #OLD:#2000 #2x procedure: 6900, 7200
	# startTime = 1350#2000
	serial = False
	if serial: #Serial
		depthDevice = DepthReader(path, framerate, startTime, serial=1, constrain=depthConstraint)
	else: #Real-time
		depthDevice = DepthReader(path, framerate, startTime, serial=0, constrain=depthConstraint)

	depthDevice.bgModel = meanIm



''' Visualization '''
cv2.namedWindow("Top")
cv2.namedWindow("Depth")
cv2.namedWindow("Signals")

'''------- Setup tracker+features -------'''
dir_ = '/Users/colin/code/pyKinectTools/data/ActionData/'
tracker = Tracker('rt_1', dir_)
markers = {
			'1':np.array([230,320,0]),
			'2':np.array([410,280,0]),
			'3_':np.array([350,50,0]),
			'4_':np.array([430,420,0]),
			'5_':np.array([330,320,0]),
			}
			# '1':np.array([240,340,0]),
			# '2':np.array([280,410,0])			
			# '1':np.array([340,240,0]),
			# '2':np.array([310,380,0])			
gss = GlobalSignalSystem(markers=markers, radius=30)

featureExt = Features(['basis'])

if 0:
	recData = np.load("/Users/colin/code/pyKinectTools/data/icu_classification_wide.npz")
	recLabels = recData['labels'].tolist()
	recFeatureNames = recData['featureNames']
	recFeatureLimits = recData['featureLimits'].tolist()
	recForest = cPickle.load(open("/Users/colin/code/pyKinectTools/data/icu_forest.pkl"))
	recSVM = cPickle.load(open("/Users/colin/code/pyKinectTools/data/icu_svm.pkl"))

# with open("icu_forest.pkl", "wb") as fid:
#     cPickle.dump(forest, fid)
# with open("icu_svm.pkl", "wb") as fid:
#     cPickle.dump(svmAll, fid)
# np.savez("icu_classification_wide.npz", featureNames=featureNames, featureLimits=featureLimits, labels=labels)

''' Voice '''
synth = AppKit.NSSpeechSynthesizer.alloc().initWithVoice_(None)

''' --------------------------------- '''
''' ------------- Main -------------- '''
''' --------------------------------- '''
depthDevice.update()
depthRaw = depthDevice.depthIm
depthRaw[depthRaw > depthConstraint[1], :] = 0
posMat = depthIm2PosIm(depthRaw)
sceneCentroid = posMat.reshape([-1,3]).mean(0)
# Office
# sceneRotation = np.asmatrix(getSceneOrientation(posMat, coordsStart=[390,375], coordsEnd=[450,465]))
# ICU
sceneRotation = np.asmatrix(getSceneOrientation(posMat,
				 coordsStart=[410,90], coordsEnd=[430,140]))
posMatNew = getTopdownMap(depthRaw, sceneRotation)

# topBounds = np.array([4000,4000,2000])
topBounds = np.array([4000,4000,depthConstraint[1]])
rezNewPos = 960


while 1:
	depthDevice.update()
	depthRaw = depthDevice.depthIm
	# colorRaw = depthDevice.colorIm
	depthRaw[depthRaw > depthConstraint[1], :] = 0
	depthRaw8 = depthDevice.depthIm8
	posMatTop = getTopdownMap(depthRaw, sceneRotation, centroid=sceneCentroid, rez=rezNewPos, bounds=topBounds)

	# bgDifference(depthDevice.bgModel)
	diff = np.array(depthDevice.bgModel, dtype=int16) - np.array(depthRaw, dtype=int16)
	diff *= (depthRaw!=0)*(depthRaw>1000)
	diffDraw = depthRaw8*(np.abs(diff) > 200)

	# imLabels, objectSlices, labelInds = extractPeople(diffDraw, minPersonPixThresh=5000, gradientFilter=False)
	imLabels, objectSlices, labelInds = extractPeople(diffDraw, minPersonPixThresh=5000, gradientFilter=True)

	'''------------ Tracking ------------'''
	if len(labelInds) > 0:
		depthRaw, com, vecs, touched = featureExt.run(depthRaw, imLabels, objectSlices, labelInds)
		ornCompare = orientationComparison(vecs)
		com_xyz = featureExt.coms_xyz

		t = time.time()
		ids = tracker.run(com_xyz, objectSlices, t, str(t), touched, vecs, ornCompare)
	else:
		com_xyz = []
		ids = tracker.run(com_xyz, [], time.time())

	''' Display COMs in image'''
	comsTopDown = []
	for i in xrange(len(com_xyz)):
		# Display COM
		c = com_xyz[i]
		c -= sceneCentroid

		comNew = np.asarray(sceneRotation*(np.asmatrix(c).T)).T[0]
		comNew += sceneCentroid
		comNew = np.asarray((comNew + topBounds/2) / topBounds*(rezNewPos-1), dtype=np.int)

		cv2.circle(posMatTop, (comNew[1], comNew[0]), radius=30, color=[255,255,255], thickness=5)
		# comsTopDown.append(np.array([comNew[1], comNew[0]]))
		comsTopDown.append(np.array(comNew))
		# Display orientation
		orn = vecs[i][2]
		ornNew = np.asarray(sceneRotation*(np.asmatrix(orn).T)).T[0]
		cv2.line(posMatTop, (int(comNew[1]-ornNew[1]*20), int(comNew[0]-ornNew[0]*20)), (int(comNew[1]+ornNew[1]*20), int(comNew[0]+ornNew[0]*20)), color=[255,255,255], thickness=5)

	'''------ Global Signal System -----'''
	gss.update(comsTopDown)
	# Add each marker position to topological map
	# for m in gss.getMarkerPos():
		# cv2.circle(posMatTop, (m[1], m[0]), radius=20, color=[150,0,0], thickness=-1)

	'''--------- Classification ---------'''
	if 0 and tracker.newSequences > 0:
		for i in xrange(tracker.newSequences, 0, -1):
			try:
				features = featureExt.extractClassificationFeatures(tracker.people[-i], recFeatureLimits)
				if features == -1:
					continue
				# Recognition
				forestActionInd = recForest.predict(features)[0]
				svmActionInd = recSVM.predict(features)[0]

				print "Action (forest, svm): ", recLabels[forestActionInd], recLabels[svmActionInd]

				# Voice output
				stringForest = "Forest: " + recLabels[forestActionInd]
				stringSVM = " SVM: " + recLabels[svmActionInd]
				synth.startSpeakingString_(stringForest + stringSVM)
			except:
				print "Error classifying this person"
		tracker.newSequences = 0

	'''--------- Pose Estimation --------'''
	''' Geodesic extrema '''
	if 0:
		for objectNum in xrange(len(labelInds)):
			posMatFull = posImage2XYZ(depthRaw8, 500, 5000)	
			posMat = posMatFull[objectSlices[objectNum]]
			for i in xrange(3):
				posMat[:,:,i] *= (imLabels[objectSlices[objectNum]]==labelInds[objectNum])
			posMat = removeNoise(posMat, thresh=500)

			''' Get regions '''

			# regionCenter = regionLabels[int(len(regionLabels)/2)][2]
			# extrema, trail, geoImg = generateKeypoints(posMat, iters=5, centroid=regionCenter, use_centroid=False)
			extrema, trail, geoImg = generateKeypoints(posMat, iters=5, use_centroid=True)

			if 1:
				regions, regionXYZ, regionLabels, edgeDict = regionGraph(posMat, pixelSize=1500)
				regionPos = [x[2] for x in regionLabels[1:]]
				regionPos.insert(0, [0,0])
				regionPos = np.array(regionPos)			
				extrema = np.array(extrema)
				geoExtrema = posMat[extrema[:,0],extrema[:,1]]
				xyz = posMat[(posMat[:,:,2]>0)*(posMat[:,:,0]!=0),:]
				geoExtrema -= xyz.mean(0)
				# skeletons, scores = pictorialScores(regionXYZ,regionPos, xyz, edgeDict, regions=regions, geoExtremaPos=extrema, geoExtrema=geoExtrema, sampleThresh=.9, gaborResponse=gaborResponse2)
				skeletons, scores = pictorialScores(regionXYZ,regionPos, xyz, edgeDict, regions=regions, geoExtremaPos=extrema, geoExtrema=geoExtrema, sampleThresh=.9)
				skeleton = skeletons[-1]

				''' Display '''
				try:
					imLab = labelSkeleton(skeleton, regionLabels, posMat[:,:,2])
				except:
					imLab = regions
				print '1'

		cv2.imshow("Depth", geoImg*1.0/geoImg.max())	

	''' Manifolds '''
	if 0 and len(labelInds) > 0:
		objectNum=0
		posMatFull = posImage2XYZ(depthRaw8, 500, 5000)	
		posMat = posMatFull[objectSlices[objectNum]]
		for i in xrange(3):
			posMat[:,:,i] *= (imLabels[objectSlices[objectNum]]==labelInds[objectNum])
		posMat = removeNoise(posMat, thresh=500)

		xyzInds = np.nonzero(posMat[:,:,2]!=0)
		# xyz = posMat[xyzInds[0], xyzInds[1]]
		# xyz -= xyz.mean(0)

		xyzInds = np.nonzero((np.gradient(posMat[:,:,2],2)[0] > 30)*(posMat[:,:,2]>0))
		xyz = posMat[xyzInds[0], xyzInds[1]]
		xyz -= xyz.mean(0)

		if xyz[0].shape > 1000:
			ptInterval=np.ceil(xyz.shape[0]/1000.0)
		else:
			ptInterval=1
		data = xyz[::ptInterval,:]
		print len(data), "data points"

		posVecs = LaplacianEigenmaps(data)

		x1= posVecs[:,0]; y1= posVecs[:,1]; z1= posVecs[:,2]
		x2= posVecs[:,3]; y2= posVecs[:,4]; #z2= posVecs[:,5]
		# x=posVecs[:,1]; y=posVecs[:,2]; z= posVecs[:,3]
		# x2= posVecs[:,4]; y2= posVecs[:,5]; z2= posVecs[:,5]		


		''' Color segments '''
		maxDim = 2
		# colorAxis = np.zeros_like(y)+maxDim
		colorAxis = np.zeros_like(posVecs[:,0])+maxDim
		for i in xrange(0,maxDim):
			colorAxis[posVecs[:,i]>0] = np.minimum(colorAxis[posVecs[:,i]>0], i)
		colorAxis += 10
		colorAxis *= 255.0/(colorAxis.max())
		colorAxis = np.round(colorAxis)

		colorMat = np.zeros_like(posMat)
		inds = xyzInds
		colorMat[inds[0][::ptInterval][::],inds[1][::ptInterval][::],0] = np.maximum(colorAxis, colorMat[inds[0][::ptInterval][::],inds[1][::ptInterval][::],0])
		# colorMat[inds[0][::ptInterval][::],inds[1][::ptInterval][::],1] = np.maximum(colorAxis.max() - colorAxis, colorMat[inds[0][::ptInterval][::],inds[1][::ptInterval][::],0])
		# imshow(colorMat[:,:,0])
		imLab = colorMat


		''' Visualize '''
		try:
			regions, regionXYZ, regionLabels, edgeDict = regionGraph(posMat, pixelSize=1500)
			allLabs = np.unique(imLab)
			sums = []
			for i in allLabs:
				sums.append(np.sum(imLab==i))
			argSums = np.argsort(sums)
			allLabs = list(allLabs[argSums])
			allLabs.remove(0)

			for i in edgeDict.keys():
				labs = unique(imLab[regions==i])
				labs = [x for x in labs if x != 0]
				# print labs
				# print labs[np.argmax([np.sum(imLab[regions==i]==x) for x in labs])]
				tmpSums = [np.sum(imLab[regions==i]==x) for x in labs]
				if len(tmpSums) > 0:
					regions[regions==i] = np.argwhere(allLabs == labs[np.argmax(tmpSums)])[0][0]+1
				else:
					regions[regions==i] = 0
		except:
			pdb.set_trace()
			print "Error"


		cv2.imshow("Depth", regions*1.0/regions.max())
	''' ---------------------------------'''

	imLabelsP = np.zeros_like(imLabels)
	lCount = 1
	for i in labelInds:
		imLabelsP += (imLabels == i)*lCount
		lCount += 1
	chart = gss.getChart()

	cv2.imshow("Depth", imLabelsP*1.0/imLabelsP.max())
	# cv2.imshow("Signals", chart/float(chart.max()))
	cv2.imshow("Signals", np.ascontiguousarray(colorRaw[:,:,0]))
	cv2.imshow("Top", posMatTop)
	# cv2.imshow("Depth", posMatTop)
	# cv2.imshow("Depth", imLab*1.0/imLab.max())

	'''Display Image'''
	ret = cv2.waitKey(10)
	if ret >= 0:
		break


# imshow(chart, interpolation="nearest")

