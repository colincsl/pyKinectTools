import os, scipy
import scipy.ndimage as nd
from pyKinectTools.utils.depthUtils import posImage2XYZ
from pyKinectTools.algs.backgroundSubtract import extractPeople, removeNoise
dataDir = '/Users/colin/data/ICU_7May2012_Close_jpg/diffDraw1/'
# dataDir = '/Users/colin/data/ICU_7May2012_Close_jpg/d1c/'
# cd /Users/colin/data/ICU_7May2012_Wide_jpg/d1c

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
sequenceFrameNames = files[0:4000:50]
imgs = []
for i in sequenceFrameNames:
    imgs.append(scipy.misc.imread(dataDir+str(i)+'.jpg'))
imgs = np.array(imgs)

''' Get posMat from individual image '''
print "Image count:", len(imgs)
for i in range(len(imgs)):
	im = imgs[i]
	objectNum = 0
	posMatFull = posImage2XYZ(im, 500, 2000)
	imLabels, objSlices, objInds = extractPeople(posMatFull[:,:,2], 10000, False)

	if len(objSlices) > 0:
		posMat = posMatFull[objSlices[objectNum]]
		for i in range(3):
			posMat[:,:,i] *= (imLabels[objSlices[objectNum]]==objInds[objectNum])

		posMat = removeNoise(posMat, thresh=500)
		xyz = posMat[(posMat[:,:,2]>0)*(posMat[:,:,0]!=0),:]

		break

