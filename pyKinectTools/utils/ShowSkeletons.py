import os
import cv2
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd
from skimage import feature
from pyKinectTools.utils.DepthUtils import world2depth
from pyKinectTools.algs.BackgroundSubtraction import *


# How to make this scale (depth) invariant?
# Normalize hog box?

textureSize = [100,50]

if 1:
	depthFolder = '/depth/'
	skelFolder = '/skel/skel/'
else:
	depthFolder = '/depth/depth/'
	skelFolder = '/skel/skel/'

def plotUsers(image, users, vis=True, device=2, backgroundModel=None, computeHog=True, computeLBP=True):

	# if backgroundModel is not None:
	# 	mask = np.abs(backgroundModel.astype(np.int16) - image) > 30
	# 	image *= mask
	ret = 0
	usersPlotted = 0
	for u in users.keys():
		xyz = users[u].com
		uvw = world2depth(xyz)

		# Only plot if there are valid coordinates (not [0,0,0])
		if uvw[0] > 0:
			if users[u].tracked:
				print users[u].userID, " tracked"
			else:
				# print ""
				pass
			# print uvw

			#Only look at box around users
			uImg = image[uvw[0]/2-textureSize[0]/2:uvw[0]/2+textureSize[0]/2, 320-(uvw[1]/2+textureSize[1]/2):320-(uvw[1]/2-textureSize[1]/2)]

			if uImg.size == 0 or uImg.max()==0:
				return 0

			uImg = np.copy(uImg).astype(np.float)
			uImg[uImg>0] -= np.min(uImg[uImg>0])
			uImg /= (np.max(uImg)/255.)

			mask = uImg > 0
			fillImage(uImg)

			# Extract features
			hogArray,hogIm = feature.hog(uImg, visualise=True)
			lbpIm = feature.local_binary_pattern(uImg, 20, 10, 'uniform')	

			hogIm *= mask
			lbpIm *= mask

			# Colorize COM
			cv2.rectangle(depthIm, tuple([320-uvw[1]/2-3, uvw[0]/2-3]), tuple([320-uvw[1]/2+3, uvw[0]/2+3]), (100))

			# Create bounding box
			# print uvw, tuple([uvw[0]/2-textureSize[0]/2, (uvw1]/2-textureSize[0]/2)]), tuple([uvw[0]/2+textureSize[1]/2,(uvw[1]/2+textureSize[1]/2)])
			# cv2.rectangle(depthIm, tuple([uvw[0]/2-textureSize[1]/2, (uvw[1]/2-textureSize[0]/2)]), tuple([uvw[0]/2+textureSize[1]/2,(uvw[1]/2+textureSize[0]/2)]), (100))
			# cv2.rectangle(depthIm, tuple([320-(uvw[1]/2-textureSize[1]/2), uvw[0]/2-textureSize[0]/2]), tuple([320-(uvw[1]/2+textureSize[1]/2), uvw[0]/2+textureSize[0]/2]), (100))

			usersPlotted += 1

	if vis is True and usersPlotted > 0:
		# Make sure windows open
		cv2.namedWindow('Depth_'+str(device))
		cv2.imshow('Depth_'+str(device), depthIm/float(depthIm.max()))
		# If there are users
		if uvw[0] > 0:
			if computeHog:
				cv2.namedWindow('HOG_'+str(device))
				# cv2.imshow('HOG_'+str(device), hogIm/float(hogIm.max()))
				cv2.imshow('HOG_'+str(device), uImg.astype(np.float)/(1.*uImg.max()))
			if computeLBP:
				cv2.namedWindow('LBP_'+str(device))
				cv2.imshow('LBP_'+str(device), lbpIm/float(lbpIm.max()))
		
		ret = cv2.waitKey(30)

	return ret


# -------------------------MAIN------------------------------------------

ret = 0
backgroundTemplates = np.empty([1,1,1])
backgroundModel = None
backgroundCount = 20;
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
		devices = [x for x in devices if x[0]!='.' and x!='tmp']
		for deviceID in devices:
			# Get filenames # Seconds
			depthTmp = os.listdir(dirs+'/'+Dir+'/'+deviceID+depthFolder)
			skelTmp = os.listdir(dirs+'/'+Dir+'/'+deviceID+skelFolder)
			# ------------ BLAHBLAHBLAH SORT THIS!!
			depthTmpSort = [int(x[-5]) if len(x)==23 else int(x[-6:-4]) for x in depthTmp]
			# depthTmpSort = ['0'+x[-5] if len(x)==23 else x[-6:-4] for x in depthTmp]			
			depthTmp = [depthTmp[i] for i in np.argsort(depthTmpSort)]
			skelTmp.sort(key=lambda x: int(x))
			# depthTmp.sort(key=lambda x: os.path.getmtime(dirs+'/'+Dir+'/'+deviceID+depthFolder+x))
			# skelTmp.sort(key=lambda x: os.path.getmtime(dirs+'/'+Dir+'/'+deviceID+skelFolder+x))
			# depthTmp.sort(key=lambda x: os.path.getctime(dirs+'/'+Dir+'/'+deviceID+depthFolder+x))
			# skelTmp.sort(key=lambda x: os.path.getctime(dirs+'/'+Dir+'/'+deviceID+skelFolder+x))			
			# depthTmp.sort()
			# skelTmp.sort()
			depthFiles.append([x for x in depthTmp if x.find('.png')>=0])
			skelFiles.append([x for x in skelTmp if x.find('.npz')>=0])
			# print depthFiles[0][0:10]
			# print skelFiles[0][0:10]

		# For each device and image
		''' Todo: seconds don't align! '''
		deviceCount = len(depthFiles)
		maxIms = np.max([len(x) for x in depthFiles])
		for i in range(maxIms):
			for d in range(deviceCount):
				if i < len(depthFiles[d]):
					depthFile = depthFiles[d][i]
					skelFile = skelFiles[d][i]
				else:
					continue
				print depthFiles[d][i]
				# Load Skeleton Data
				data = np.load(dirs+'/'+Dir+'/'+devices[d]+skelFolder+skelFile)
				users = data['users'].tolist()
				data.close()
				coms = [users[x].com for x in users.keys() if users[x].com[2] > 0.0]
				# if len(users.keys()) == 0
				# if backgroundTemplates.shape[2] == backgroundCount and len(coms) <= 1:
				# 	continue
				jointCount = 0
				for u in users.items():
					# print "Joint size:", len(u[1].jointPositions.keys())
					if len(u[1].jointPositions.keys()) > 0:
						jointCount = len(u[1].jointPositions.keys())
				# if jointCount == 0:
				# 	continue

				# Load Image
				depthIm = sm.imread(dirs+'/'+Dir+'/'+devices[d]+depthFolder+depthFile)
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

				mask = None
				if backgroundModel is None:
					backgroundModel = depthIm.copy()
					backgroundTemplates = depthIm[:,:,np.newaxis].copy()
				else:
					mask = np.abs(backgroundModel.astype(np.int16) - depthIm)
					mask[depthIm < 500] = 0

					depthBG = depthIm.copy()
					# depthBG[~mask] = 0
					if backgroundTemplates.shape[2] < backgroundCount or np.random.rand() < bgPercentage:
						# mask = np.abs(backgroundTemplates[:,:,0].astype(np.int16) - depthIm)# < 20
						backgroundTemplates = np.dstack([backgroundTemplates, depthBG])
						backgroundModel = backgroundTemplates.sum(-1) / np.maximum((backgroundTemplates>0).sum(-1), 1)
						# backgroundModel = nd.maximum_filter(backgroundModel, np.ones(2))
					if backgroundTemplates.shape[2] > backgroundCount:
						backgroundTemplates = backgroundTemplates[:,:,1:]

					# depthIm[mask] = 0




				cv2.namedWindow("a")
				# cv2.imshow("a", backgroundModel.astype(np.float) /3000.0)
				cv2.imshow("a", depthIm.astype(np.float) /4000.0)
				try:
					pass
					# cv2.imshow("a", mask.astype(np.float)/mask.max())
					# cv2.imshow("a", mask.astype(np.uint8)*255)
				except:
					pass
				ret = cv2.waitKey(50)
				# # ret = 1
				imLabels, objectSlices, labelInds = extractPeople(depthIm, minPersonPixThresh=500, gradientFilter=True)

				imTmp = np.zeros_like(imLabels)
				for l in labelInds:
					imTmp = np.maximum(imTmp, imLabels==l)

				# depthIm[:, 1:-2] *= imTmp
				# ret = plotUsers(depthIm, users, device=devices[d], backgroundModel=backgroundModel)
				# try:
				# 	ret = plotUsers(depthIm, users, device=devices[d], backgroundModel=backgroundModel)
				# except:
				# 	print "Error plotting"

				if ret > 0:
					break					
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
