
import numpy as np, cv, cv2
import time, os
from copy import deepcopy
from random import randint

import pyKinectTools.utils.DepthReader
from pyKinectTools.utils.DepthUtils import *


class Tracker:
	def __init__(self, name="PeopleTracker", dir_="."):
		self.people = []
		self.people_open = []
		self.name = name #+ "_" + str(int(time.time()))
		self.dir = dir_
		self.newSequences = 0

	def run(self, coms_, slices=[], time=[], depthFilename=[], touches=[], basis=[], ornCompare=[]):

		coms = deepcopy(list(coms_))
		comLabels = []
		deleteTimeThresh = 3
		movingCount = 3

		if len(coms) > 0 and len(self.people_open) > 0:
			distMat = np.zeros([len(coms), len(self.people_open)])
			for j in xrange(len(coms)):
				for pp in xrange(len(self.people_open)):
					p = self.people_open[pp]
					#x,y,z dist
					distMat[j,pp] = np.sqrt(np.sum((np.array(coms[j])-np.array(p['moving_com'][-1]))**2))
					# distMat[j,pp] = np.sqrt(np.sum((np.array(coms[j])-np.array(p['com'][-1]))**2))

			distMatSort = np.argsort(distMat, axis=1)
			
			deleteLst = []		
			for i in xrange(len(coms)):
				prevInd = distMatSort[i,0]
				# if prevInd == i and distMat[i, prevInd] < 400:
				if distMat[i, prevInd] < 500:
					self.people_open[prevInd]['com'].append(coms[i])
					moving_com = np.mean(np.vstack([self.people_open[prevInd]['moving_com'][-movingCount:],coms[i]]), axis=0)
					self.people_open[prevInd]['moving_com'].append(moving_com) #moving avg
					self.people_open[prevInd]['slice'].append(slices[i])
					self.people_open[prevInd]['time'].append(time)
					self.people_open[prevInd]['filename'].append(depthFilename)
					if len(basis)>0: self.people_open[prevInd]['basis'].append(basis[i])
					if len(ornCompare)>0: self.people_open[prevInd]['ornCompare'].append(ornCompare[i])
					touches = [y for x, y in zip(touches, range(len(touches))) if i in x]
					if len(touches) > 0:
						if 'touches' not in self.people_open[prevInd]:
							self.people_open[prevInd]['touches'] = []
						self.people_open[prevInd]['touches'].append([touches, len(self.people_open[prevInd]['filename'])])
					distMat[:,prevInd] = 9999
					distMatSort = np.argsort(distMat, axis=1)
					deleteLst.append(i)
					comLabels.append(self.people_open[prevInd]['label'])
			deleteLst.sort(reverse=True)
			for i in deleteLst:
						coms.pop(i)

		''' If there are any new, unaccounted for people, add them '''
		for j in xrange(len(coms)):
			print "New person"
			i=0
			while(i in comLabels):
				i += 1
			comLabels.append(i)			

			self.people_open.append({'com':[coms[j]], 'moving_com':[coms[j]], 'slice':[slices[j]], 
				'time':[time], 'filename':[depthFilename], 'label':i, 'basis':[basis[i]], 'ornCompare':[ornCompare[i]]})

		'''Convert old people to 'overall' instead of current'''
		deleteLst = []
		for j in xrange(len(self.people_open)):
			p = self.people_open[j]
			# import pdb
			# pdb.set_trace()
			if p['time'][-1] < time-deleteTimeThresh:
				pAvg = np.array(np.mean(p['com'], 0)) #centroid over time
				timeDiff = p['time'][-1]-p['time'][0]
				self.people.append({"data":p, "start":p['time'][0], 
								"elapsed":timeDiff,
								"com":pAvg, 'label':p['label']})
				print "Person left"
				self.newSequences += 1
				deleteLst.append(j)
		
		deleteLst.sort(reverse=True) # remove last nodes first
		for j in deleteLst:
			self.people_open.pop(j)

		if len(coms) == 0:
			return []

		return comLabels

	def finalize(self):
		deleteLst = []
		for j in range(len(self.people_open)):
			p = self.people_open[j]
			pAvg = np.array(np.mean(p['com'], 0))
			timeDiff = p['time'][-1]-p['time'][0]
			self.people.append({"data":p, "start":p['time'][0], 
							"elapsed":timeDiff,
							"com":pAvg, 'label':p['label']})
			deleteLst.append(j)
		deleteLst.sort(reverse=True) # remove last nodes first
		for j in deleteLst:
			self.people_open.pop(j)

		startTime = 99999
		endTime = 0
		for p in self.people:
			if p['start'] < startTime:
				startTime = p['start']
			if p['start']+p['elapsed'] > endTime:
				endTime = p['start']+p['elapsed']

		meta = {'start':startTime, 'end':endTime, 'elapsed':endTime-startTime,
				'count':len(self.people)}
		os.chdir(self.dir)
		print "Tracker saved to ", (self.dir+self.name)
		np.savez(self.name, data=self.people, meta=meta)




#------------------------------------------------------------------------

def showLabeledImage(data, ind_j, dir_, rgb=0):
	cv.NamedWindow("a")
	if rgb:
		file_ = data['data']['filename'][ind_j]
		img = DepthReader.getRGBImage(dir_+file_[:file_.find(".depth")]+".rgb")
		# imgD=img[:,::-1, :]
		imgD=img
	else:
		img = DepthReader.getDepthImage(dir_+data['data']['filename'][ind_j])
		imgD = DepthReader.constrain(img, 500, 5000)
		imgD = np.dstack([imgD, imgD, imgD])

	# com_xyz =  np.array(data['data']['com'])
	com_xyz =  np.array(data['data']['moving_com'])
	com = list(world2depth(com_xyz).T) #COM in image coords
	# Add previous location markers
	r = 5
	for jj in range(ind_j):
		s = (slice(com[jj][0]-r, com[jj][0]+r),slice(com[jj][1]-r, com[jj][1]+r))
		imgD[s[0], s[1], 0] = 100
		imgD[s[0], s[1], 1:3] = 0
	# Add current location marker
	r = 10
	s = (slice(com[ind_j][0]-r, com[ind_j][0]+r),slice(com[ind_j][1]-r, com[ind_j][1]+r))
	colorInd = 2#randint(0, 2)
	if colorInd==0: # Vary the colors every new person
		r=255; g=0; b=0
	elif colorInd==1:
		r=0; g=255; b=0
	elif colorInd==2:
		r=0; g=0; b=255				
	imgD[s[0], s[1], 0] = r
	imgD[s[0], s[1], 1] = g
	imgD[s[0], s[1], 2] = b	
	# Add touch boxes
	touchEnabled = 1
	if touchEnabled:
		rad = 13
		cv2.putText(imgD, "Touches", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), thickness=1)
		for i in range(2):
			cv2.rectangle(imgD, (60-rad, 40*i+60-rad), (60+rad, 40*i+60+rad), [255, 255, 255])
	if 'touches' in data['data'].keys():
		rad = 10
		times = [x[1] for x in data['data']['touches']]
		if any(np.equal(times,ind_j)):
			for i in data['data']['touches'][np.argwhere(np.equal(times,ind_j))[0]][0]:
				cv2.circle(imgD, (60, 40*i+60), rad, [0, 255, 0], thickness=4)
	# Print time on screen
	time_ = data['data']['time'][ind_j]
	hours = int(time_ / 3600)
	minutes = int(time_ / 60) - hours*60
	seconds = int(time_) - 60*minutes - 3600*hours
	text = str(hours) + "hr " + str(minutes) + "min " + str(seconds) + "s"
	cv2.putText(imgD, text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
	# Print duration to screen
	text = "Dur: " + str(data['elapsed'])
	cv2.putText(imgD, text, (450, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
	#Show image
	cv2.imshow("a", imgD)
	cv2.waitKey(1)


def labelData(filename, dir_='/Users/colin/data/ICU_7March2012_Head/', speed=3):
	playData(filename, dir_, speed, label=True)

def playData(filename, dir_='/Users/colin/data/ICU_7March2012_Head/', speed=3, label=False, filterbox=None, startFrame=0):

	data_raw = np.load(filename)
	p = data_raw['data']
	m = data_raw['meta']
	name = filename[:filename.find(".")] + "_labeled"
	p = filterEvents(p)
	print "Action count:", len(p)

	cv.NamedWindow("a")
	imgD = np.zeros([480,640, 3]) # Dislay each image.

	for ii in range(startFrame, len(p)): # For each segmented person
		data = p[ii] # person data
		# if data['elapsed'] > 10: # filter really short segments
		if 1:#len(data['data']['time']) > 5:
			com = np.array([data['com']])
			# com_uv = list(world2depth(com).T)[0] #COM in image coords
			play = 0#1
			for j in xrange(0, len(data['data']['time']), speed):

				showLabeledImage(data, j, dir_)

				# Get label
				if label and (play or (j > len(data['data']['time'])-speed)):
					# lab = raw_input("Label (Play: p; Prev Err: pe [label]; Save: s): ")
					lab = raw_input(("Label frame "+str(ii)+": "))
					if lab == "p":
						play = 0
					elif lab[:2] == "pe":
						errorFrames.append(ii)
						data['label'] = lab[3:]
					elif lab[0] == "s":
						np.savez(name, data=p, meta=m)
						play = 0
					else:
						data['label'] = lab
						break

				time.sleep(.01)
			# imgD[:,:,:] = 255
			# cv2.imshow("a", imgD)
			# ret = cv2.waitKey(1)
			# time.sleep(.05)

			if label:
				np.savez(name, data=p, meta=m)


def filterEvents(data, filterbox=[]):
	# data = [x for x in data if len(x['data']['time']) > 10] #5
	data = [x for x in data if x['elapsed'] > 10] #5

	# # Stuff on this side of the bed gets errased
	if 0:
		filterbox = [[0,240],[300, 680],[0, 4000]] # x, y, z
		com = np.array([x['com'] for x in data])
		com = list(world2depth(com).T) #COM in image 
		data = [data[y] for x,y in zip(com, range(len(data))) if not (filterbox[0][0] < x[0] < filterbox[0][1] and filterbox[1][0] < x[1] < filterbox[1][1] and filterbox[2][0] < x[2] < filterbox[2][1])]

	return data

#------------------------------------------------------------------------


