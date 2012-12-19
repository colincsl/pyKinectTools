import os
import cPickle as pickle
import cv2
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd
from skimage import feature
from pyKinectTools.utils.DepthUtils import world2depth
from pyKinectTools.algs.BackgroundSubtraction import *
from pyKinectTools.utils.RealtimeReader import *

''' Get features relative to objects '''

''' Find temporal sub-second offset '''

# textureSize = [100,50]
textureSize = [50,50]
Cam_OFFSETS = [0, 18, 13]
MS_OFFSET = -20
allFeatures_perUser = {}

if 1:
	depthFolder = '/depth/'
	skelFolder = '/skel/'
else:
	depthFolder = '/depth/depth/'
	skelFolder = '/skel/skel/'


def computeFeatures_perSecond(imageFiles, userFiles, dev=-1):
	''' images is a set of 1 second worth of images '''

	features_perSecond = {}
	features_perUser = {}
	hogs = {}

	# Go through each user for all frames
	for i in xrange(len(imageFiles)):
		# data = np.load(dirs+'/'+Dir+'/'+'device_'+str(dev)+skelFolder+userFiles[i])
		data = pickle.load(dirs+'/'+Dir+'/'+'device_'+str(dev)+skelFolder+userFiles[i])
		users = data['users'].tolist()
		data.close()
		time = [int(x.split('_')[-2])*100 + int(format(x.split('_')[-1][:x.split('_')[-1].find('.')])) for x in depthTmp]

		for u in users.keys():
			frameFeat = {}
			xyz = users[u]['com']
			uvw = world2depth(xyz)

			# Only plot if there are valid coordinates (not [0,0,0])
			if uvw[0] > 0:

				frameFeat['com'] = xyz
				frameFeat['time'] = users[u].timestamp
				if u not in features_perUser.keys():
					features_perUser[u] = [frameFeat]
				else:
					features_perUser[u].append(frameFeat)

				if u not in hogs or i == int(len(imageFiles)/2):

					image = sm.imread(dirs+'/'+Dir+'/'+devices[dev]+depthFolder+imageFiles[i])
					cv2.namedWindow("Pre-HOG")
					# import pdb; pdb.set_trace()
					cv2.imshow("Pre-HOG", image.astype(np.float)/5000)
					cv2.waitKey(10)
					image = np.array(image, dtype=np.uint16)

					#Only look at box around users
					uImg = image[uvw[0]/2-textureSize[0]/2:uvw[0]/2+textureSize[0]/2, (uvw[1]/2-textureSize[1]/2):(uvw[1]/2+textureSize[1]/2)]
					if uImg.size == 0 or uImg.max()==0:
						continue

					uImg = np.ascontiguousarray(uImg)
					uImg.resize(textureSize)
					uImg = np.copy(uImg).astype(np.float)
					uImg[uImg>0] -= np.min(uImg[uImg>0])
					uImg /= (np.max(uImg)/255.)

					# mask = uImg > 0
					fillImage(uImg)

					# Extract features
					hogArray = feature.hog(uImg, visualise=False)
					# hogArray *= mask.flatten()
					# lbpIm = feature.local_binary_pattern(uImg, 20, 10, 'uniform')					
					# lbpIm *= mask

					hogs[u] = hogArray


	#Summarize frames
	for u in features_perUser.keys():
		pos = np.mean([x['com'] for x in features_perUser[u]], 0)
		var = np.std([x['com'] for x in features_perUser[u]], 0)
		time = np.array([features_perUser[u][0]['time']])
		# import pdb; pdb.set_trace()
		hog = hogs[u]#features_perUser[u][int(len(features_perUser[u])/2)]['hog']#.reshape([-1,1])
		dev = np.array([dev])

		features_perSecond[u] = np.concatenate([time, dev, pos, var, hog])


	return features_perSecond






def computeFeatures_perFrame(image, users, device=2, computeHog=True, computeLBP=False, vis=False):

	features = {}

	# Head, Left hand, right hand
	bodyIndicies = [0, 5, 8] 

	for u in users.keys():
		xyz = users[u]['com']
		uvw = world2depth(xyz)
		hogs = []

		# Only plot if there are valid coordinates (not [0,0,0])
		if uvw[0] > 0 and users[u]['tracked']:

			''' HOG at each Body positions '''
			for i in bodyIndicies:
				pt = world2depth(users[u]['jointPositions'][users[1]['jointPositions'].keys()[i]])
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



def plotUsers(image, users, vis=True, device=2, backgroundModel=None, computeHog=True, computeLBP=False):

	# if backgroundModel is not None:
	# 	mask = np.abs(backgroundModel.astype(np.int16) - image) > 30
	# 	image *= mask
	ret = 0
	usersPlotted = 0
	uvw = [-1]
	# hogPositions = ['head', 'hand_l', 'hand_r']
	bodyIndicies = [0, 5, 8] # See SkeletonUtils.py
	hogs = []
	for u in users.keys():
		if users[u]['tracked']:
			xyz = users[u]['com']
			uvw = world2depth(xyz)

			# Only plot if there are valid coordinates (not [0,0,0])
			if uvw[0] > 0:
				if users[u]['tracked']:

					''' Body positions '''

					for i in bodyIndicies:
						pt = world2depth(users[u]['jointPositions'][users[1]['jointPositions'].keys()[i]])
						uImg = image[pt[0]/2-textureSize[0]/2:pt[0]/2+textureSize[0]/2, (pt[1]/2-textureSize[1]/2):(pt[1]/2+textureSize[1]/2)]

						if uImg.size == 0 or uImg.max()==0:
							return

						uImg = np.ascontiguousarray(uImg)
						uImg.resize(textureSize)

						uImg = np.copy(uImg).astype(np.float)
						uImg[uImg>0] -= np.min(uImg[uImg>0])
						# from IPython import embed
						# embed()

						uImg = np.minimum(uImg, 100)
						# uImg /= (np.max(uImg)/255.)

						fillImage(uImg)
						# hogArray = feature.hog(uImg, visualise=False)

						hogArray, hogIm = feature.hog(uImg, visualise=True)
						cv2.namedWindow("hog_"+str(i))
						cv2.imshow("hog_"+str(i), hogIm)
						# cv2.imshow("hog_"+str(i), uImg)
						
						ret = cv2.waitKey(10)
						hogs.append(hogArray)


					''' Whole body '''
					#Only look at box around users
					# uImg = image[uvw[0]/2-textureSize[0]/2:uvw[0]/2+textureSize[0]/2, (uvw[1]/2-textureSize[1]/2):(uvw[1]/2+textureSize[1]/2)]
					# # uImg = image[uvw[0]/2-textureSize[0]/2:uvw[0]/2+textureSize[0]/2, 320-(uvw[1]/2+textureSize[1]/2):320-(uvw[1]/2-textureSize[1]/2)]			

					# if uImg.size == 0 or uImg.max()==0:
					# 	return 0

					# uImg = np.copy(uImg).astype(np.float)
					# uImg[uImg>0] -= np.min(uImg[uImg>0])
					# uImg /= (np.max(uImg)/255.)

					# mask = uImg > 0
					# fillImage(uImg)

					# # Extract features
					# hogArray,hogIm = feature.hog(uImg, visualise=True)
					# lbpIm = feature.local_binary_pattern(uImg, 20, 10, 'uniform')	

					# hogIm *= mask
					# lbpIm *= mask

					# Colorize COM
					cv2.rectangle(depthIm, tuple([uvw[1]/2-3, uvw[0]/2-3]), tuple([uvw[1]/2+3, uvw[0]/2+3]), (4000))

					# Create bounding box
					# print uvw, tuple([uvw[0]/2-textureSize[0]/2, (uvw1]/2-textureSize[0]/2)]), tuple([uvw[0]/2+textureSize[1]/2,(uvw[1]/2+textureSize[1]/2)])
					# cv2.rectangle(depthIm, tuple([uvw[0]/2-textureSize[1]/2, (uvw[1]/2-textureSize[0]/2)]), tuple([uvw[0]/2+textureSize[1]/2,(uvw[1]/2+textureSize[0]/2)]), (100))
					# cv2.rectangle(depthIm, tuple([320-(uvw[1]/2-textureSize[1]/2), uvw[0]/2-textureSize[0]/2]), tuple([320-(uvw[1]/2+textureSize[1]/2), uvw[0]/2+textureSize[0]/2]), (100))

					# Plot skeleton
					if vis:
						w = 3
						# print "Joints: ", len(u.jointPositions)
						for j in users[u]['jointPositions'].keys():
							pt = world2depth(users[u]['jointPositions'][j])
							depthIm[pt[0]/2-w:pt[0]/2+w, pt[1]/2-w:pt[1]/2+w] = 4000                                                        

					usersPlotted += 1

	if vis is True and usersPlotted >= 0:
		# Make sure windows open
		cv2.namedWindow('Depth_'+str(device))
		cv2.imshow('Depth_'+str(device), depthIm/float(depthIm.max()))
		# If there are users
		if uvw[0] > 0:
			# if computeHog:
				# cv2.namedWindow('HOG_'+str(device))
				# cv2.imshow('HOG_'+str(device), hogIm/float(hogIm.max()))
				# cv2.imshow('HOG_'+str(device), uImg.astype(np.float)/(1.*uImg.max()))
			if computeLBP:
				cv2.namedWindow('LBP_'+str(device))
				cv2.imshow('LBP_'+str(device), lbpIm/float(lbpIm.max()))
		
		ret = cv2.waitKey(30)

	return ret

def format(x):
	if len(x) == 1:
		# print x
		# return '-'+x+'9999999'
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

ret = 0
backgroundTemplates = np.empty([1,1,1])
backgroundModel = None
backgroundCount = 20
bgPercentage = .05

hourDirs = os.listdir('.')
hourDirs = [x for x in hourDirs if x[0]!='.']
hourDirs.sort(key=lambda x: int(x))

for dirs in hourDirs: # Hours
	minuteDirs = os.listdir(dirs)
	minuteDirs = [x for x in minuteDirs if x[0]!='.']
	minuteDirs.sort(key=lambda x: int(x))
	for Dir in minuteDirs: # Minutes

		if Dir[0] == '.': # Prevent from reading hidden files
			continue

		depthFiles = []
		skelFiles = []

		# For each available device:
		devices = os.listdir(dirs+'/'+Dir)
		devices = [x for x in devices if x[0]!='.' and x.find('tmp')<0]
		devices.sort()
		for deviceID in devices:
			# Get filenames # Seconds
			depthTmp = os.listdir(dirs+'/'+Dir+'/'+deviceID+depthFolder)
			skelTmp = os.listdir(dirs+'/'+Dir+'/'+deviceID+skelFolder)
			''' Sort files '''
			# depthTmp = [x for x in depthTmp if len(x.split('_')[-1][:x.split('_')[-1].find('.')])==2]
			# Extract times as number
			# tmpSort = [int(x.split('_')[-3])*100 + int(format(x.split('_')[-1][:x.split('_')[-1].find('.')])) for x in depthTmp]			

			tmpSort = [int(x.split('_')[-3])*100 + int(format(x.split('_')[-2])) for x in depthTmp]			
			depthTmp = np.array(depthTmp)[np.argsort(tmpSort)].tolist()
			depthFiles.append([x for x in depthTmp if x.find('.png')>=0])

			tmpSort = [int(x.split('_')[-4])*100 + int(format(x.split('_')[-3])) for x in skelTmp]			
			skelTmp = np.array(skelTmp)[np.argsort(tmpSort)].tolist()			
			skelFiles.append([x for x in skelTmp if x.find('.dat')>=0])

		# 	dev = int(deviceID[-1])

		# 	# Add features
		# 	try:
		# 		features = computeFeatures(depthFiles[dev], skelFiles[dev], dev=dev)
		# 		for u in features.keys():
		# 			if u in allFeatures_perUser:
		# 				allFeatures_perUser[u].append(features[u])
		# 			else:
		# 				allFeatures_perUser[u] = [features[u]]
		# 		print features.keys(), allFeatures_perUser.keys()
		# 	except:
		# 		print "Error making features"

		# ret = cv2.waitKey(10)
		# if ret >= 0:
		# 	break

		# if 1:
		# 	continue


		# For each device and image	

		# deviceCount = len(depthFiles)
		# maxIms = np.max([len(x) for x in depthFiles])
		# for i in range(maxIms):
			# for d in range(deviceCount):
				# if i < len(depthFiles[d]):
				# 	depthFile = depthFiles[d][i]
				# 	skelFile = skelFiles[d][i]
				# else:
				# 	continue
				# print depthFiles[d][i]
				# if len(depthFiles[d][i]) < 24:
				# 	continue
			# depthFile = depthFiles[d][i]
			# skelFile = skelFiles[d][i]		

		for dev, depthFile in multiCameraTimeline(depthFiles):

			skelFile = 'skel_'+depthFile[6:-4]+'_.dat'

			# Load Skeleton Data
			# data = np.load(dirs+'/'+Dir+'/'+devices[dev]+skelFolder+skelFile)
			try:
				with open(dirs+'/'+Dir+'/'+devices[dev]+skelFolder+skelFile, 'rb') as inFile:
					users = pickle.load(inFile)
			except:
				print "No user file:", skelFile
				continue
			# users = data['users'].tolist()
			# data.close()
			coms = [users[x]['com'] for x in users.keys() if users[x]['com'][2] > 0.0]


			jointCount = 0
			for i in users.keys():
				user = users[i]
				# if user['jointPositions'][user['jointPositions'].keys()[0]] != -1:
					# print user['jointPositionsConfidence']
					# jointCount = 1
			print depthFile

			# if backgroundTemplates.shape[2] == backgroundCount and len(coms) <= 1:
			# if len(coms) < 1:	
				# continue

			# Load Image
			depthIm = sm.imread(dirs+'/'+Dir+'/'+devices[dev]+depthFolder+depthFile)
			depthIm = np.array(depthIm, dtype=np.uint16)

			# Background model
			# fillImage(depthIm)
			# mask = None
			# if backgroundModel is None:
			# 	backgroundModel = depthIm.copy()
			# 	backgroundTemplates = depthIm[:,:,np.newaxis].copy()
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


			# ''' Background model #2 '''
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


			# cv2.namedWindow("a")
			# cv2.imshow("a", backgroundModel.astype(np.float) /5000.0)
			# cv2.imshow("a", depthIm.astype(np.float) /5000.0)
			# try:
				# pass
				# cv2.imshow("a", mask.astype(np.float)/mask.max())
			# 	cv2.imshow("a", mask.astype(np.uint8)*255)
			# except:
			# 	pass
			# ret = cv2.waitKey(20)
			# # ret = 1
			# imLabels, objectSlices, labelInds = extractPeople(depthIm, minPersonPixThresh=500, gradientFilter=True)

			# imTmp = np.zeros_like(imLabels)
			# for l in labelInds:
			# 	imTmp = np.maximum(imTmp, imLabels==l)

			# depthIm[:, 1:-2] *= imTmp
			features = computeFeatures_perFrame(depthIm, users, device=devices[dev], vis=True)
			if features is not None:
				for k in features.keys():
					if k not in allFeatures_perUser.keys():
						allFeatures_perUser[k] = [features[k]]
					else:
						allFeatures_perUser[k].append(features[k])
			# ret = plotUsers(depthIm, users, device=devices[dev], backgroundModel=backgroundModel)
			# try:
			# 	ret = plotUsers(depthIm, users, device=devices[dev], backgroundModel=backgroundModel)
			# except:
			# 	print "Error plotting"

				# if ret > 0:
					# break					
			if ret > 0:
				break
		if ret > 0:
			break
	if ret > 0:
		break



# imLabels, objectSlices, labelInds = extractPeople(depthIm, minPersonPixThresh=500, gradientFilter=True)








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