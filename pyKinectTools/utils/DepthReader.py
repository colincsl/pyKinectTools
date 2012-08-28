


import os, time, sys
import numpy as np
import cv, cv2
from math import floor
# import SkelPlay as SR
import pyKinectTools.utils.SkeletonUtils as SR
# from pyKinectTools.utils.DepthUtils import *
from pyKinectTools.algs.BackgroundSubtraction import constrain

def getFolderTime(folderName):
	# Folder name is of format hr-min-sec
	return (int(folderName[0:2])*60*60 + int(folderName[3:5])*60 + int(folderName[6:8]))

def getFileTime(fileName, fileNames):
	if len(fileNames) < 4:
		return getFolderTime(fileName)
	# fileNames.sort
	start = int(fileNames[0][fileNames[0].find("_")+1:fileNames[0].find(".")])
	end   = int(fileNames[-1][fileNames[-1].find("_")+1:fileNames[-1].find(".")])
	current = int(fileName[fileName.find("_")+1:fileName.find(".")])
	diff = end-start

	return getFolderTime(fileName) + (current-start) / diff

def getFileListTimes(fileNames):
	if len(fileNames) < 4:
		currentTime = getFolderTime(fileNames[0])
		# print currentTime, fileNames
		return [[currentTime, fileNames]]
	start = float(fileNames[0][fileNames[0].find("_")+1:fileNames[0].find(".")])
	end   = float(fileNames[-1][fileNames[-1].find("_")+1:fileNames[-1].find(".")])
	diff = end-start
	times = []

	for f in fileNames:
		current = float(f[f.find("_")+1:f.find(".")])
		currentTime = getFolderTime(f) + float(float(current-start) / diff)
		if len(times) == 0 or times[-1][0] != currentTime:
			currentList = [f]
			times.append([currentTime, currentList])
		else:
			times[-1][1].append(f)

	return times

# def constrain(img):
# 	min_ = np.min(img[np.nonzero(img)])
# 	max_ = img.max() #/2
# 	img[np.nonzero(img)] -= min_
# 	img = np.clip(img, min_, max_)		
# 	img -= min_
# 	# print max_, min_
# 	img = np.array((img / ((max_-min_)/256.0)), dtype=np.uint8)
# 	img = 256 - img
# 	img = np.array(img, dtype=np.uint8)
# 	return img

# def constrain(img, mini=-1, maxi=-1): #500, 4000
# 	if mini == -1:
# 		min_ = np.min(img[np.nonzero(img)])
# 	else:
# 		min_ = mini
# 	if maxi == -1:
# 		max_ = img.max()
# 	else:
# 		max_ = maxi

# 	img = np.clip(img, min_, max_)
# 	img -= min_
# 	if max_ == 0:
# 		max_ = 1
# 	img = np.array((img * (255.0 / (max_-min_))), dtype=np.uint8)

# 	return img

def getDepthImage(depthFilename):
	x = open(depthFilename).read()
	return np.fromstring(x, dtype=np.uint16, sep=" ").reshape([480, 640])[:,::-1]	

def getRGBImage(rgbFilename):	
	imgData = np.fromfile(rgbFilename, dtype=np.uint8)
	return imgData.reshape([480, 640, 3])

class DepthReader:
	def __init__(self, path, framerate=30, timeOffset=0, clockTime=0, cameraNumber=0, viz=0, vizSkel=0, skelsEnabled=0, serial=0, constrain=[]):
		self.path = path
		self.framerate = framerate
		self.timeOffset = timeOffset
		self.cameraNumber = cameraNumber
		self.viz = viz
		self.vizSkel = vizSkel
		self.constrain = constrain
		self.skelsEnabled = skelsEnabled
		self.windowName = "DEPTH" + str(cameraNumber)
		if viz:
			cv.NamedWindow(self.windowName)
		
		os.chdir(self.path)
		self.DirNames = os.listdir('.')
		self.DirNames = [x for x in self.DirNames if (x[0] != '.')] # make sure it's not a hidden file
		self.DirNames.sort()

		self.startDirTime = (getFolderTime(self.DirNames[0]))
		self.endDirTime = getFolderTime(self.DirNames[-1])
		if clockTime == 0:
			self.startTime = time.time()
		else:
			self.startTime = clockTime

		self.serial = serial
		self.allPaths = []
		if serial:
			for i in (self.DirNames):
				os.chdir(i)
				files = os.listdir('.')
				# for i in range(len(files)-1, -1, -1):
				# 	del files[i]
				fileTmp = []
				for j in files:
					if j[-5:] != 'depth':
						continue
					fileTmp.append(j)
					filesTmp2 = [i+"/"+x for x in files if x[:x.find(".")] == fileTmp[-1][:fileTmp[-1].find(".")] and x != fileTmp[:-1]]

					self.allPaths.append([filesTmp2])
				os.chdir('..')

	def reset(self, time):
		self.startTime = time.time()

	def run(self, initTime=0):
		os.chdir(self.path)

		if self.serial:
			files = self.allPaths.pop(0)[0]
			self.currentDirTime = getFolderTime(files[0][:files[0].find("/")])
			tmp = ((self.currentDirTime - self.startDirTime))
			tmpMin = int(tmp/60.0)
			tmpSec = int(tmp - tmpMin*60)
			tmp2 = (self.endDirTime-self.startDirTime)
			tmpTotMin = int(floor(tmp2/60.0))
			tmpTotSec = int(tmp2 - tmpTotMin*60)
			self.timeMin = tmpMin
			self.timeSec = tmpSec			
		else:
			self.currentDirTime = 0
			if initTime == 0:
				initTime = time.time()
			currentTime = (initTime - self.startTime) * (self.framerate / 30.0) + self.timeOffset
			# print currentTime
			# currentTime = (initTime - self.startTime) * (self.framerate / 30.0)

			# Get current directory
			for i in xrange(len(self.DirNames)):
				if i+1 < len(self.DirNames):
					if currentTime < (getFolderTime(self.DirNames[i+1])-self.startDirTime):
						self.currentDirTime = getFolderTime(self.DirNames[i])
						os.chdir(self.path+self.DirNames[i])
						# self.DirNames.pop(i)
						break
					else:
						continue
				else:
					self.currentDirTime = getFolderTime(self.DirNames[i])
					os.chdir(self.path+self.DirNames[i])
					# self.DirNames.pop(i)
					break

			if self.currentDirTime > 0:
				filenames = os.listdir('.')
				times = getFileListTimes(filenames)

				# pdb.set_trace()

				tmp = ((self.currentDirTime - self.startDirTime))
				tmpMin = int(tmp/60.0)
				tmpSec = int(tmp - tmpMin*60)
				tmp2 = (self.endDirTime-self.startDirTime)
				tmpTotMin = int(floor(tmp2/60.0))
				tmpTotSec = int(tmp2 - tmpTotMin*60)
				self.timeMin = tmpMin
				self.timeSec = tmpSec
				# print "C" + str(self.cameraNumber) + "	Time: " + str(tmpMin) + " min " + str(tmpSec) + " sec of " + str(tmpTotMin) + " min " + str(tmpTotSec) + " sec"

				# Get current time
				files = []
						
				#Get closest fileset
				for t in xrange(len(times)):
					if currentTime > times[t][0] - self.startDirTime:
						if (t+1)<len(times):
							if currentTime < times[t+1][0] - self.startDirTime:
								files = times[t][1]
							else:
								continue
						else:
							files = times[t][1]
					else:
						break

		self.getFrameData(files)

	def update(self, initTime=0):
		self.run(initTime)

	def getFrameData(self, files):
		depthFilename = []
		rgbFilename = []
		skelFilename = []

		# pdb.set_trace()

		# Get filenames
		if len(files) > 0:
			for i in files:
				if i[-5:] == 'depth':
					depthFilename = i
					self.depthFilename = depthFilename
				if i[-3:] == 'rgb':
					rgbFilename = i
					self.rgbFilename = rgbFilename
				if self.skelsEnabled:
					if i[-4:] == 'skel':
						skelFilename = i
						self.skelFilename = skelFilename

			displayUsers = []
			skels = []
			if len(skelFilename) > 0:
				users = SR.readUserData(skelFilename)
				skels = SR.readSkeletonData(skelFilename)
				for i in users:
					displayUsers.append(users[i]['Img'])

				#eliminate bad skeleton data (joint data is linked to old data)
				# pdb.set_trace()
				deleteInd = []
				for i in users.keys():
					# print users[i]['Img']
					if users[i]['Img'][2] == 0:
						deleteInd.append(i)
				deleteInd.sort(reverse=1)
				for i in deleteInd:
					# print "Del", i
					del users[i]
					del skels[i]
					del displayUsers[i-1]

			if len(depthFilename) > 0:
				## 1
				# depthRaw = open(depthFilename, 'rb').read().split()
				# depthRaw = np.fromfile(depthFilename, dtype=np.uint16, sep=" ")
				# self.depthData = np.array(depthRaw, dtype=int).reshape([480,640])[:,::-1]
				# self.depthData = self.depthData[:,::-1]
				## 2
				# self.depthData = np.fromfile(depthFilename, dtype=np.uint16, sep=" ").reshape([480, 640])[:,::-1]
				# self.depthDataRaw = self.depthData
				## 3
				x = open(depthFilename).read()
				self.depthDataRaw = np.fromstring(x, dtype=np.uint16, sep=" ").reshape([480, 640])[:,::-1]	
				self.depthIm = self.depthDataRaw
				if self.constrain != []:
					self.depthIm8 = constrain(self.depthIm, self.constrain[0], self.constrain[1])

				# print "User count: ", len(displayUsers)
				# print displayUsers
				if self.viz:
					self.depthData = constrain(self.depthDataRaw)
					try:
						for i in displayUsers:
							if i[0] != 0:
								for x in xrange(-10,10):
									for j in xrange(-1,1):
										self.depthData[480 - i[1]+j, 640 - i[0]+x] = 30
								for y in xrange(-10, 10):
									for j in xrange(-1,1):
										self.depthData[480 - i[1]+y, 640 - i[0]+j] = 30
					except:
						print "Error adding cross at", i

				# self.depthData = cv.fromarray(np.array(self.depthData, dtype=np.uint8))
				# print "Skels: ", len(skels)
				# print skels
				if skels != [] and self.vizSkel:
					self.depthData = cv.fromarray(np.array(self.depthData, dtype=np.uint8))
					self.depthData = SR.displaySkeleton_CV(self.depthData, skels)
				if self.viz:
					self.depthData = cv.fromarray(np.array(self.depthData, dtype=np.uint8))
					cv2.imshow(self.windowName, np.array(self.depthData, dtype=np.uint8))

			if len(rgbFilename) > 0:
				imgData = np.fromfile(rgbFilename, dtype=np.uint8)
				imgData = imgData.reshape([480, 640, 3])
				if self.viz:
					cv2.imshow("RGB", imgData)

			cv.WaitKey(1)


