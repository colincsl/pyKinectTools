import os, scipy
import scipy.ndimage as nd
from pyKinectTools.utils.depthUtils import posImage2XYZ
from pyKinectTools.algs.backgroundSubtract import extractPeople, removeNoise
from pyKinectTools.algs.geodesicSkeleton import *
from pyKinectTools.algs.pictorialStructures import *

dataDir = '/Users/colin/data/ICU_7May2012_Wide_jpg/diffDraw1/'

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
sequenceFrameNames = files[410:440]
# sequenceFrameNames = files[410:1101]
imgs = []
for i in sequenceFrameNames:
    imgs.append(scipy.misc.imread(dataDir+str(i)+'.jpg'))
imgs = np.array(imgs)

''' Get posMat from individual image '''
im = imgs[6]
objectNum = 0
posMatFull = posImage2XYZ(im, 500, 4000)
imLabels, objSlices, objInds = extractPeople(posMatFull[:,:,2], 10000, False)
posMat = posMatFull[objSlices[objectNum]]
for i in range(3):
	posMat[:,:,i] *= (imLabels[objSlices[objectNum]]==objInds[objectNum])

posMat = removeNoise(posMat, thresh=500)
xyz = posMat[(posMat[:,:,2]>0)*(posMat[:,:,0]!=0),:]

''' Get geodesic extrema '''
from pyKinectTools.algs.geodesicSkeleton import *
extrema, trail, geoImg = generateKeypoints(np.ascontiguousarray(posMat), xyz, 10)



regions, regionXYZ, regionLabels = regionGraph(posMat)
labelGraphImage(regionLabels)
pictorialScores(regionXYZ,xyz, edgeDict)