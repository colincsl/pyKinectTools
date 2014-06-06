'''
Tools to help calculating and displaying features
'''

from sklearn.decomposition import FastICA, NMF, DictionaryLearning
from matplotlib.pylab import subplot, imshow
import numpy as np


def saveFeatures(featureDict):

	featureList = featureDict.items()
	labels = [x[0] for x in featureList[0]]

	data = []
	for i in range(len(featureList)):
		data.append([x[1] for x in featureList[i]])

	np.savez("/media/Data/allFeatures_tmp", labels=labels, data=data)

def loadFeatures(filename):
	file_ = np.load(filename)
	labels = file_['labels']
	data = file_['data']

	return labels, data

# from sklearn.decomposition import FastICA
def learnICADict(features, components=25):
	

	icaHOG = FastICA(n_components=components)
	icaHOF = FastICA(n_components=components)

	icaHOG.fit(np.array([x['hog'] for x in features]).T)
	icaHOF.fit(np.array([x['hof'] for x in features]).T)

	hogComponents = icaHOG.components_.T
	hofComponents = icaHOF.components_.T

	return hogComponents, hofComponents

# from sklearn.decomposition import FastICA
def learnNMFDict(features, components=25):
	from sklearn.decomposition import NMF

	nmfHOG = NMF(n_components=components)
	nmfHOF = NMF(n_components=components)

	nmfHOG.fit(np.array([x['hog'] for x in features]).T)
	nmfHOF.fit(np.array([x['hof'] for x in features]).T)

	hogComponents = icaHOG.components_.T
	hofComponents = icaHOF.components_.T

	return hogComponents, hofComponents	

if 0:
	from sklearn.decomposition import DictionaryLearning
	dicHOG = DictionaryLearning(25)
	dicHOG.fit(hogs)


def displayComponents(components):
	
	sides = ceil(np.sqrt(len(components)))
	for i in range(len(components)):
		subplot(sides, sides, i+1)
		imshow(hog2image(components[i], imageSize=[24,24],orientations=4))

	sides = ceil(np.sqrt(components.shape[1]))
	for i in range(components.shape[1]):
		subplot(sides, sides, i+1)
		imshow(hog2image(components[:,i], imageSize=[24,24],orientations=4))

