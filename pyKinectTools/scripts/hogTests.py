import sys
from sklearn import svm
# from sklearn import svm
import scipy.ndimage as nd
import scipy.io
from scipy.io.matlab import loadmat
from pyKinectTools.utils.HOGUtils import *

sys.path.append("/Users/colin/libs/pyvision/build/lib.macosx-10.7-intel-2.7/")
from vision import features # pyvision library

import pyKinectTools.data as ktData
dataFolder = ktData.__file__
dataFolder = dataFolder[:dataFolder.find("__init__")]

#-------

# Get training data
trainingFeatures = scipy.io.loadmat(dataFolder+'bodyPart_HOGFeatures.mat')
trainingDepths = scipy.io.loadmat(dataFolder+'bodyPart_DepthImgs.mat')

bodyLabels = [x for x in trainingFeatures if x[0]!='_' and x!='other']
featureCount = np.sum([len(trainingFeatures[x]) for x in bodyLabels if x!='other'])
vectorLen = trainingFeatures['r_hand'][0].flatten().shape[0]

allFeatures = np.empty([featureCount, vectorLen])
allFeaturesLab = np.empty([featureCount], dtype=str)
allFeaturesLabNum = np.empty([featureCount],dtype=uint8)

tmp_i = 0
for lab,i in zip(bodyLabels, range(len(bodyLabels))):
	for feat in trainingFeatures[lab]:
		f = feat.flatten()
		f[f<0] = 0

		allFeatures[tmp_i,:] = f
		allFeaturesLab[tmp_i] =  lab
		allFeaturesLabNum[tmp_i] = i+1
		tmp_i += 1

# display all heads

faces = trainingDepths['face'][0][0]
for i in range(1, len(trainingDepths['face'])):
	imTmp = trainingDepths['face'][i][0]
	if imTmp.shape[0] != 0:
		faces = np.dstack([faces, imTmp])


''' Gabor experiment '''
if 0:
	angles = range(0, 108*2, 90/5)
	gabors = generateGabors(angles, [10,10], .5)

	convs = []
	for i in range(len(angles)):
		convs.append(nd.convolve(imTmp, gabors[:,:,i]))
	convs = np.array(convs)

''' Display all faces '''
import cv, cv2
cv2.namedWindow("Face", cv2.CV_WINDOW_AUTOSIZE)
for i in range(len(trainingDepths['face'])-1):
	# imshow(trainingDepths['face'][i][0])
	tmpImg = np.asarray(trainingDepths['face'][i][0], dtype=float)
	## Normalize the image. Min min/max within standard deviations to help eliminate the background
	if not np.all(tmpImg == 0):
		imMean = tmpImg.mean()
		imSTD = tmpImg.std()
		min_ = tmpImg[tmpImg > imMean-1*imSTD].min()
		tmpImg -= min_
		max_ = tmpImg[tmpImg < imMean-min_+1*imSTD].max()
		tmpImg[tmpImg>max_]=max_
		tmpImg[tmpImg<0]=max_
		tmpImg /= np.float(max_/255.0)
	# tmpImg = np.minimum(tmpImg, 255)

	tmpImg = np.asarray(tmpImg, dtype=np.uint8)
	cv2.imshow("Face", tmpImg)
	ret = cv2.waitKey(50)
	if ret >= 0:
		break
cv2.destroyWindow("Face")

allbodyparts = ['face', 'l_hand', 'r_hand']
hogs = {}
hogRes = 8
for part in allbodyparts:
	hogs[part] = []

	bodypart = part
	''' Get HOG features '''
	for i in range(len(trainingDepths[bodypart])-1):
		tmpImg = np.asarray(trainingDepths[bodypart][i][0], dtype=float)
		## Normalize the image. Min min/max within standard deviations to help eliminate the background
		if not np.all(tmpImg == 0):
			imMean = tmpImg.mean()
			imSTD = tmpImg.std()
			min_ = tmpImg[tmpImg > imMean-1*imSTD].min()
			tmpImg -= min_
			max_ = tmpImg[tmpImg < imMean-min_+1*imSTD].max()
			tmpImg[tmpImg>max_]=max_
			tmpImg[tmpImg<0]=max_
			tmpImg /= np.float(max_/255.0)	
			tmpImg = np.asarray(tmpImg, dtype=np.uint8)

			tmp = np.dstack([tmpImg, tmpImg, tmpImg])
			hogs[part].append(features.hog(tmp, hogRes))


from pyKinectTools.utils.HOGUtils import *
hogList = list(hogs['face'])
for p in hogs['l_hand']:
	hogList.append(p)
# hogList.append(hogs['l_hand'])
# hogList = np.array(hogList)
hogVals = np.empty([len(hogList), len(hogList[0].flatten())])
for i in range(len(hogList)):
	# hogList[i] = hogList[i].flatten()
	hogVals[i,:] = hogList[i][0].flatten()[0]
hogLabels = np.zeros(len(hogs['face'])+len(hogs['l_hand']))
# hogLabels[0:len(hogs['face'])]+=1
hogLabels[len(hogs['face']):]+=1

# im = HOGpicture(hogList[1], hogRes)
# imshow(im, interpolation='nearest')

svm_ = svm.SVC(kernel='poly', probability=True, C=1, degree=2, gamma=1)
# svm_ = svm.SVC(kernel='rbf', probability=True, nu=.7)#, C=1)#, )
svm_.fit(hogVals, hogLabels)

score = svm_.score(hogVals, hogLabels)
probs = svm_.predict_proba(hogVals)
print score

from sklearn.ensemble import RandomForestClassifier	
from sklearn.ensemble import ExtraTreesClassifier

forest = ExtraTreesClassifier(n_estimators=50, compute_importances=True, n_jobs=7, bootstrap=True, random_state=0, max_features=1)
forest.fit(hogVals, hogLabels)
score = forest.score(hogVals, hogLabels)
print score


'''--------------------------------------------'''

'''Catagorical SVMs'''
#'''Train'''
# svms = []
# for feat,i in zip(allFeatures, range(len(allFeaturesLab))):
# for i in bodyLabels:
	# labels = np.zeros(featureCount)
labels = allFeaturesLabNum
svm_ = svm.SVC(probability=True, C=100)
svm_.fit(allFeatures, labels)
# svms.append(deepcopy(svm_))
#'''Test training'''
svmPredict = np.empty([featureCount])
for feat,i in zip(allFeatures, range(featureCount)):
	# for j in xrange(len(bodyLabels)):
		# svmPredict[i,j] = svm_.predict_proba(feat)[0][1]
	svmPredict[i] = svm_.predict(feat)


for k in range(len(extrema)-1):
	tmpBoxRot = allBoxesRot[k][5:35, 5:35]
	tmpBoxD = np.dstack([tmpBoxRot, tmpBoxRot, tmpBoxRot])

	if tmpBoxRot.shape[0] == 30 and tmpBoxRot.shape[1] == 30:
		f = features.hog(tmpBoxD, 4).flatten()
		print svm_.predict(f),svm_.predict_proba(f), extrema[k]

	scores = np.empty(featureCount)
	for j in xrange(featureCount):
		scores[j] = svms[j].predict_proba(f)[0][1]
	scores /= np.sum(scores)



'''Exemplar SVMs'''
#'''Train'''
svms = []
for feat,i in zip(allFeatures, range(len(allFeaturesLab))):
	labels = np.zeros(featureCount)
	labels[i] = allFeaturesLabNum[i]
	svm_ = svm.SVC(probability=True, C=100)
	svm_.fit(allFeatures, labels)
	svms.append(deepcopy(svm_))
#'''Test training'''
svmPredict = np.empty([featureCount,featureCount])
for feat,i in zip(allFeatures, range(featureCount)):
	for j in xrange(featureCount):
		svmPredict[i,j] = svms[j].predict_proba(feat)[0][1]


	# labels = np.zeros(len(allFeaturesLab))
	# labels[i] = 1
	# # svm_ = svm.NuSVC(nu=.2, probability=True)
	# svm_ = svm.SVC(probability=True)
	# svm_.fit(allFeatures, labels)


# Problem: isn't there a way to make it invariant on the input size (the training data bounding box size is modified for fitting)
#2:face; #3:r_hand; 5:l_hand

for k in range(len(extrema)):
	tmpBoxRot = allBoxesRot[k][10:30, 10:30]
	tmpBoxD = np.dstack([tmpBoxRot, tmpBoxRot, tmpBoxRot])

	f = features.hog(tmpBoxD, 4)[0].flatten()
	scores = np.empty(featureCount)
	for j in xrange(featureCount):
		scores[j] = svms[j].predict_proba(f)[0][1]
	scores /= np.sum(scores)
	
	t = np.vstack([allFeaturesLabNum, scores]).T
	max_ = 0
	argmax_ = -1
	for c in [2,3,5]:
		tmp = np.sum(t[t[:,0]==c,1]) / np.sum(t[:,0]==c)
		if tmp > max_:
			max_ = tmp
			argmax_ = c
	print argmax_


	print bodyLabels[allFeaturesLabNum[argmax(scores)]-1], argmax(scores), extrema[k]





for k in range(9):
	print k
	tmpBoxRot = allBoxesRot[k]
	# tmpBoxRot = np.nan_to_num(tmpBoxRot)
	tmpBoxRot[tmpBoxRot>0] -= tmpBoxRot[tmpBoxRot>0].min()
	tmpBoxRot[tmpBoxRot>0] /= tmpBoxRot.max()
	# tmpBoxRot[tmpBoxRot>0] = np.log(tmpBoxRot[tmpBoxRot>0])
	tmpBoxD = np.dstack([tmpBoxRot, tmpBoxRot, tmpBoxRot])
	f = features.hog(tmpBoxD, 4)

	# figure(1); subplot(3,3,k+1)
	figure(1); imshow(tmpBoxRot)

	# figure(2); subplot(3,3,k+1)
	# imshow(f[:,:,i-1])

	im = HOGpicture(f, 4)
	figure(2); imshow(im, interpolation='nearest')


