import os, scipy, time
import scipy.ndimage as nd
from pyKinectTools.utils.DepthUtils import posImage2XYZ
from pyKinectTools.algs.PictorialStructures import *
from pyKinectTools.algs.BackgroundSubtraction import extractPeople, removeNoise
dataDir = '/Users/colin/data/ICU_7May2012_Close_jpg/diffDraw1/'
# dataDir = '/Users/colin/data/ICU_7May2012_Close_jpg/d1c/'

import cv, cv2

global bodyPos
global partIndex
global posMat

AllBodyPos = []
bodyPos = []
bodyTimes = []
partIndex = 0


def onclick(event):

	# global regions
	global posMat
	global bodyPos
	global partIndex

	# print "h0", posMat[int(event.ydata), int(event.xdata),:]
	print partIndex
	bodyPos.append(posMat[int(event.ydata), int(event.xdata),:])

	partIndex += 1
	if partIndex == 5:#len(bodyPos.keys()):
		partIndex = 0


'''#################### Load Images #########################'''

'''
imgs = array of many images
im = specific image
posMat = 3-dimensional array of XYZ positions at each pixel
xyz = list of points
'''

files = os.listdir(dataDir)
files = [int(x[0:-4]) for x in files if x[0]!='.']
files = np.sort(files)
# sequenceFrameNames = files[0:4000:50]
# sequenceFrameNames = files[7200:7230]
sequenceFrameNames = files[::500]
imgs = []
for i in sequenceFrameNames:
    imgs.append(scipy.misc.imread(dataDir+str(i)+'.jpg'))
imgs = np.array(imgs)

''' ------- MAIN --------- '''
print "Start"

t = 0

if t > 0:
	if bodyPos == -1 or len(bodyPos) == 7:
		AllBodyPos.append(bodyPos)
		bodyTimes.append(sequenceFrameNames[t-1])
	else:
		print "Error. Redoing the last frame."
		t -= 1
else:
	AllBodyPos = []

bodyPos = []

im = imgs[t]
objectNum = 0
posMatFull = posImage2XYZ(im, 500, 2000)
imLabels, objSlices, objInds = extractPeople(posMatFull[:,:,2], 10000, True)
if len(objInds)==0:
	print"No humans"
	bodyPos = -1
else:
	posMat = posMatFull[objSlices[objectNum]]
	for i in range(3):
		posMat[:,:,i] *= (imLabels[objSlices[objectNum]]==objInds[objectNum])

	posMat = removeNoise(posMat, thresh=500)

	fig = figure(2)
	fig.canvas.mpl_connect('button_press_event', onclick)
	ax = fig.add_subplot(111)	
	ax.imshow(posMat[:,:,2])

t += 1


''' ------- \MAIN --------- '''

''' ---- Compute stats ----- '''

# relativePos = [x[2]-x[1], x[]]] for x in AllBodyPos]

labels = ["r_shoulder", "r_arm", "r_hand",
		  "l_shoulder", "l_arm", "l_hand"]
relDistsIndiv = [
			[x[1]-x[0] for x in AllBodyPos if x != -1 and np.any(x[1] != 0)],
			[x[2]-x[1] for x in AllBodyPos if x != -1 and np.any(x[2] != 0)],
			[x[3]-x[2] for x in AllBodyPos if x != -1 and np.any(x[3] != 0)],
			[x[4]-x[0] for x in AllBodyPos if x != -1 and np.any(x[4] != 0)],
			[x[5]-x[4] for x in AllBodyPos if x != -1 and np.any(x[5] != 0)],
			[x[6]-x[5] for x in AllBodyPos if x != -1 and np.any(x[6] != 0)]
			]

relDists = [np.mean(x, 0) for x in relDistsIndiv]
absDists = [np.sqrt(np.sum(x**2)) for x in relDists]
relStds = [np.std(x,0) for x in relDistsIndiv]
absStds = [np.std(x) for x in relDistsIndiv]

scipy.savez("labeledSkels_Every500.npz",
			relDistsIndiv=relDistsIndiv,
			relDists=relDists,
			absDists=absDists,
			relStds=relStds,
			absStds=absStds,
			times=bodyTimes, 
			dataDir=dataDir)


