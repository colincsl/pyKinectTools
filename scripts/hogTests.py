import sys
from scikits.learn import svm
import scipy.ndimage as nd
from scipy.io.matlab import loadmat
from pyKinectTools.HOGUtils import *

sys.path.append("/Users/colin/libs/pyvision/build/lib.macosx-10.7-intel-2.7/")
from vision import features # pyvision library

#-------


# Get training data
trainingFeatures = io.loadmat('bodyPart_HOGFeatures.mat')
trainingDepths = io.loadmat('bodyPart_DepthImgs.mat')

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


