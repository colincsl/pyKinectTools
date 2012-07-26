'''
This file extracts depth images for all annotated joint data in the chalearn datasets.

Colin Lea
June 2012
'''

import csv
import cv, cv2
from copy import deepcopy
import scipy.io as io
import random

import sys
sys.path.append("/Users/colin/libs/pyvision/build/lib.macosx-10.7-intel-2.7/")
from vision import features

calcHOG = 1

S_R_HAND = 1
S_L_HAND = 2
S_HEAD = 3
S_R_SHOULDER = 4
S_L_SHOULDER = 5
S_R_ELBOW = 6
S_L_ELBOW = 7

SKELETON = [S_R_HAND, S_L_HAND, S_HEAD, S_R_SHOULDER, S_L_SHOULDER, S_R_ELBOW, S_L_ELBOW]

fid = csv.reader(open('/Users/colin/data/chalearn/body_parts.csv'))
tmpData = []

for i in fid:
    tmpData.append(i)

#must translate annotated position by [-40, -40]

data = {}
for i in range(1, len(tmpData)):
	devSet = tmpData[i][0]
	vidSet = tmpData[i][1]
	ident = devSet
	vidIdent = 'K_' + vidSet + '.avi'
	# ident = str(devSet)
	frame = int(tmpData[i][2])
	joint = int(tmpData[i][3])
	uncertain = int(tmpData[i][4])
	left = int(tmpData[i][5])
	top  =int(tmpData[i][6])
	width = int(tmpData[i][7])
	height = int(tmpData[i][8])
	pos = (left-40,top-40)
	size = (width, height)

	if not data.has_key(ident):
		data[ident] = {}
	if not data[ident].has_key(vidIdent):
		data[ident][vidIdent] = {}
		data[ident][vidIdent]['pos'] = {}
		data[ident][vidIdent]['size'] = {}
		data[ident][vidIdent]['uncertain'] = {}

	data[ident][vidIdent]['frame'] = frame
	data[ident][vidIdent]['pos'][joint] = pos
	data[ident][vidIdent]['size'][joint] = size
	data[ident][vidIdent]['uncertain'][joint] = uncertain


im = np.empty([240,320])

folder = '/Users/colin/data/chalearn/devel/'


bodyPartDepths = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
bodyPartFeatures = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
mapping = ["r_hand","l_hand","face","r_shoulder","l_shoulder","r_elbow","l_elbow"]

otherDepths = []
otherFeatures = []

#Save all depth data
for setNum in data.keys():
	for vidNum in data[setNum].keys():

		cam = cv2.VideoCapture(folder+setNum+"/"+vidNum)
		frame = data[setNum][vidNum]['frame']
		cam.set(cv.CV_CAP_PROP_POS_FRAMES,frame)
		f, im = cam.retrieve()
		im = im*(im<im.mean())

		pos = data[setNum][vidNum]['pos']
		size = data[setNum][vidNum]['size']
		uncertain = data[setNum][vidNum]['uncertain']
		boxSize = 60
		for b in range(1,8):
			if uncertain[b] == 0 and size[b][0] > 0:
				size[b] = (boxSize,boxSize) #!!
				pos[b] = (pos[b][0] + int(size[b][0]/2) - boxSize/2,
						  pos[b][1] + int(size[b][1]/2) - boxSize/2)
				# cv2.rectangle(im, pos[b], (pos[b][0]+size[b][0], pos[b][1]+size[b][1]), [200,100,100])
				box = im[pos[b][1]:pos[b][1]+size[b][1], pos[b][0]:pos[b][0]+size[b][0], 2]
				# boxMean = int(box.mean())
				# boxMax = box.max()
				# box[box<boxMean] = box.max()
				# box[box>=boxMean] -= box[box>0].min()
				bodyPartDepths[b].append(deepcopy(box))
				if calcHOG and box.shape[0] > 0 and box.shape[1] > 0 and box.shape[0] == boxSize and box.shape[1] == boxSize:
					# tmpBox = np.dstack([box[:,:,2], box[:,:,2], box[:,:,2]])
					tmpBoxD = np.dstack([box,box,box])
					f = features.hog(tmpBoxD, 4)
					bodyPartFeatures[b].append(deepcopy(f))

					# im2 = HOGpicture(f, 4)
					# figure(2); imshow(im, interpolation='nearest')

					# for i in range(5):
					# 	posNew = [pos[b][0]+random.randint(30,50), pos[b][1]+random.randint(30,50)]
					# 	box = im[pos[b][1]:pos[b][1]+size[b][1], pos[b][0]:pos[b][0]+size[b][0], 2]
					# 	if calcHOG and box.shape[0] > 0 and box.shape[1] > 0 and box.shape[0] == 30 and box.shape[1] == 30:
					# 		otherDepths.append(deepcopy(box))
					# 		tmpBoxD = np.dstack([box,box,box])
					# 		f = features.hog(tmpBoxD, 4)
					# 		otherFeatures.append(deepcopy(f))



		# figure(1); subplot(2,4,b)
		# imshow(box)
		# figure(2); imshow(im)
	
		# f, im = cam.read()
		# imshow(im)

		if 1:
			# cv2.imshow("1", im*(im<150))
			cv2.imshow("1", tmpBoxD)
			ret = cv2.waitKey(100)

			if ret >= 0:
				break


io.savemat("bodyPart_DepthImgs.mat", {
									mapping[0]:bodyPartDepths[1],
									mapping[1]:bodyPartDepths[2],
									mapping[2]:bodyPartDepths[3],
									mapping[3]:bodyPartDepths[4],
									mapping[4]:bodyPartDepths[5],
									mapping[5]:bodyPartDepths[6],
									mapping[6]:bodyPartDepths[7],
									"other":otherDepths
	})
io.savemat("bodyPart_HOGFeatures.mat", {
									mapping[0]:bodyPartFeatures[1],
									mapping[1]:bodyPartFeatures[2],
									mapping[2]:bodyPartFeatures[3],
									mapping[3]:bodyPartFeatures[4],
									mapping[4]:bodyPartFeatures[5],
									mapping[5]:bodyPartFeatures[6],
									mapping[6]:bodyPartFeatures[7],
									"other":otherFeatures
	})


