import os, scipy
import scipy.ndimage as nd
from pyKinectTools.utils.DepthUtils import posImage2XYZ
from pyKinectTools.algs.BackgroundSubtraction import extractPeople, removeNoise
from pyKinectTools.algs.GeodesicSkeleton import *
from pyKinectTools.algs.PictorialStructures import *
from pyKinectTools.algs.STIP import *

# dataDir = '/Users/colin/data/ICU_7May2012_Wide_jpg/diffDraw1/'
dataDir = '/Users/colin/data/ICU_7May2012_Close_jpg/diffDraw1/'
# dataDir = '/Users/colin/data/ICU_7May2012_Close_jpg/d1c/'

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
# sequenceFrameNames = files[610:690]
# sequenceFrameNames = files[2000:2101]
# sequenceFrameNames = files[1200:1300]
sequenceFrameNames = files[::500]
# sequenceFrameNames = files[7200:7230] #:7300
imgs = []
for i in sequenceFrameNames:
    imgs.append(scipy.misc.imread(dataDir+str(i)+'.jpg'))
imgs = np.array(imgs)

''' Get posMat from individual image '''
# t=7
t=13
im = imgs[t]
objectNum = 0
# posMatFull = posImage2XYZ(im, 500, 1250)
posMatFull = posImage2XYZ(im, 500, 2000)
imLabels, objSlices, objInds = extractPeople(posMatFull[:,:,2], 10000, True)
if len(objInds)!=0:
	t += 1
assert len(objInds)!=0, "Error: No objects"
	
posMat = posMatFull[objSlices[objectNum]]
for i in range(3):
	posMat[:,:,i] *= (imLabels[objSlices[objectNum]]==objInds[objectNum])
posMat = removeNoise(posMat, thresh=500)
xyz = posMat[(posMat[:,:,2]>0)*(posMat[:,:,0]!=0),:]

''' Get geodesic extrema '''
t1 = time.time()
regions, regionXYZ, regionLabels, edgeDict = regionGraph(posMat, pixelSize=1500)
regionPos = [x[2] for x in regionLabels[1:]]
regionPos.insert(0, [0,0])
regionPos = np.array(regionPos)
regionCenter = regionLabels[int(len(regionLabels)/2)][2]
extrema, trail, geoImg = generateKeypoints(posMat, iters=10, centroid=regionCenter, use_centroid=False)

''' Gabors '''
im = posMat[:,:,2]
# grad = np.diff(im)
angles = range(0, 180, 45)
gabors = generateGabors(angles, [30,30], 1.0)
convs = []
for i in range(len(angles)):
	convs.append(nd.convolve(im, gabors[:,:,i]))
	# convs.append(nd.convolve(grad, gabors[:,:,i]))
convs = np.array(convs)
mask = im>0
imOut = convs.mean(0)
# imOut *= mask[:,1:]
imOut *= mask
grad = np.gradient(np.asarray(imOut, dtype=float), 1)
mag = np.sqrt(grad[0]**2+grad[1]**2)
imOut -= imOut.min()
imOut /= imOut.max()
# gaborResponse=(1.0-imOut)
gaborResponse=(imOut)
gaborResponse[gaborResponse==1] = 0
gaborResponse2 = np.zeros_like(gaborResponse)
regionCount = regions.max()
for i in range(1, regionCount+1):
	# gaborResponse2[regions==i] = gaborResponse[regions==i].mean()
	gaborResponse2[regions==i] = gaborResponse[regionPos[i][0],regionPos[i][1]]


''' Pictorial Structures '''
extrema = np.array(extrema)
geoExtrema = posMat[extrema[:,0],extrema[:,1]]
geoExtrema -= xyz.mean(0)
skeletons, scores = pictorialScores(regionXYZ,regionPos, xyz, edgeDict, regions=regions, geoExtremaPos=extrema, geoExtrema=geoExtrema, sampleThresh=.9, gaborResponse=gaborResponse2)
skeleton = skeletons[-1]
print time.time() - t1

''' Display '''
figure(2); imshow(geoImg)
imLab = labelSkeleton(skeleton, regionLabels, posMat[:,:,2])
figure(3); imshow(imLab)
# labelGraphImage(regionLabels)
t+=1



# ''' Gabors '''
# # im = np.diff(posMat[:,:,2])
# im = posMat[:,:,2]
# angles = range(0, 180, 45/2)
# # gabors = generateGabors(angles, [20,20], 1)
# gabors = generateGabors(angles, [20,20], 1.0)
# # gabors = generateGabors(angles, [5,5], .75)
# convs = []
# for i in range(len(angles)):
# 	convs.append(nd.convolve(im, gabors[:,:,i]))
# convs = np.array(convs)
# imOut = convs.max(0)#*mask
# grad = np.gradient(np.asarray(imOut, dtype=float), 1)
# mag = np.sqrt(grad[0]**2+grad[1]**2)
# imOut -= imOut.min()
# imOut /= imOut.max()
