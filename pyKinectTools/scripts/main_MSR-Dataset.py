import cv, cv2
import numpy as np
import scipy.misc as sm
import os, time
from pyKinectTools.utils.SkeletonUtils import display_MSR_skeletons
from pyKinectTools.utils.DepthUtils import world2depth
from pyKinectTools.dataset_readers.MSR_DailyActivities import read_MSR_depth_ims, read_MSR_color_ims, read_MSR_skeletons, read_MSR_labels
from pyKinectTools.algs.BackgroundSubtraction import extract_people
from pyKinectTools.algs.HistogramOfOpticalFlow import getFlow, hof

from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.decomposition import DictionaryLearning,MiniBatchDictionaryLearning
from sklearn import svm

import pickle
from IPython import embed



# def compute_skeletal_gradients()


def compute_features(name, vis=False, person_rez=[144,72]):
	'''
	---Parameters---
	filename : base filename for depth/color/skel
	vis: turn visualize on?
	person_rez : all hog/hof features should be the same dimensions, so we resize the people to this resolution
	---Return---
	features: features in a dictionary
	'''

	''' Get filenames '''
	depth_file = name + "depth.bin"
	color_file = name + "rgb.avi"
	skeleton_file = name + "skeleton.txt"
	''' Read data from each video/sequence '''
	depthIms, maskIms = read_MSR_depth_ims(depth_file)
	colorIms = read_MSR_color_ims(color_file)
	skels_world, skels_im = read_MSR_skeletons(skeleton_file)

	dataset_features = {'hog':[], 'hof':[], 'skel_image':[], 'skel_world':[]}
	framecount = np.minimum(depthIms.shape[0], colorIms.shape[0])
	grayIm_prev = None

	''' View all data'''
	for frame in xrange(framecount):
		depth = depthIms[frame]
		mask = maskIms[frame]
		color = colorIms[frame]
		# Skeleton in world (w) and image (i) coordinates
		skel_w = skels_world[frame]
		skel_i = world2depth(skel_w, rez=[240,320])

		''' Calculate hogs '''
		grayIm = (rgb2gray(color) * 255).astype(np.uint8)
		hogIm = np.zeros_like(depth)

		person_mask, bounding_boxes, labels = extract_people(grayIm, mask>0)
		rez = grayIm[bounding_boxes[0]].shape

		hog_input_im = sm.imresize(grayIm[bounding_boxes[0]], person_rez)
		hogData, hogImBox = hog(hog_input_im, orientations=4, visualise=True)
		
		hogIm[bounding_boxes[0]] = sm.imresize(hogImBox, [rez[0],rez[1]])
		# hogIm[bounding_boxes[0]] = hogImBox
		hogIm *= person_mask

		''' Calculate HOF '''
		hofIm = np.zeros_like(depth)
		if grayIm_prev is None:
			grayIm_prev = np.copy(grayIm)
			continue
		else:
			flow = getFlow(grayIm_prev[bounding_boxes[0]], grayIm[bounding_boxes[0]])
			rez = flow.shape
			bounding_boxes = (bounding_boxes[0][0], bounding_boxes[0][1], slice(0,2))

			hof_input_im = np.dstack([sm.imresize(flow[0], [person_rez[0],person_rez[1]]),
										sm.imresize(flow[1], [person_rez[0],person_rez[1]])])

			hofData, hofImBox = hof(hof_input_im, orientations=5, visualise=True)
			hofIm[bounding_boxes[:2]] = sm.imresize(hofImBox, [rez[0],rez[1]])
			hofIm *= person_mask
		grayIm_prev = np.copy(grayIm)


		''' Add features '''
		dataset_features['hog'] += [hogData]
		dataset_features['hof'] += [hofData]
		dataset_features['skel_image'] += [skel_i]
		dataset_features['skel_world'] += [skel_w]

		''' Plot skeletons on color image'''
		if vis:
			color = display_MSR_skeletons(color, skel_i)

			''' Visualization '''
			cv2.imshow("Depth", depth/float(depth.max()))
			cv2.imshow("HOG", hogIm/float(hogIm.max()))
			cv2.imshow("RGB", color)
			cv2.imshow("RGB masked", color*(mask[:,:,None]>0))
			cv2.imshow("HOF", hofIm/float(hofIm.max()))
			ret = cv2.waitKey(10)

			if ret >= 0:
				break

	print "Done calculating ", name
	return dataset_features


def main_calculate_features():
	''' Get all appropriate files in this folder '''
	files = os.listdir('.')
	base_names = [f[:12] for f in files]
	#Remove bad filenames
	base_names = [f for f in base_names if f[0]!='.' and f[0]=='a']
	base_names = np.unique(base_names)

	''' Initialize feature vectors '''
	dataset_features = {}

	''' Compute features '''
	start_time = time.time()
	try:
		from joblib import Parallel, delayed
		print "Computing with multiple threads"
		data = Parallel(n_jobs=3)( delayed(compute_features)(n) for n in base_names )
		for n,i in zip(base_names, range(len(base_names))):
			dataset_features[n] = data[i]
	except:
		print "Computing with single thread"
		for n in base_names:
			dataset_features = compute_features(n, vis=False)
	print 'Total time:', time.time() - start_time			


	with open('MSR_Features_hog-hof-skel_%f.dat'%time.time(), 'wb') as outfile:
	    pickle.dump(dataset_features, outfile, protocol=pickle.HIGHEST_PROTOCOL)

	embed()

def normalize_skeleton(data):
	data = np.array(data)
	data -= data.mean(0)
	data /= data.var(0)
	return data.reshape([-1,60])

def create_dictionaries(n_codewords=20):
	dataset_features = np.load('MSR_Features_hog-hof-skel1360423760.27.dat')
	hogs = []
	hofs = []
	skels = []
	for n in dataset_features.keys():
		hogs +=	dataset_features[n]['hog']
		hofs +=	dataset_features[n]['hof']
		skels += [normalize_skeleton(dataset_features[n]['skel_world'])]

	''' Input should be features[n_samples, n_features] '''
	hogs = np.vstack(hogs)
	hofs = np.vstack(hofs)
	skels = np.vstack(skels)

	hog_dict = MiniBatchDictionaryLearning(n_codewords, n_jobs=-1, verbose=True, transform_algorithm='lasso_lars')
	hog_dict.fit(hogs)
	hof_dict = MiniBatchDictionaryLearning(n_codewords, n_jobs=-1, verbose=True, transform_algorithm='lasso_lars')
	hof_dict.fit(hofs)
	skels_dict = MiniBatchDictionaryLearning(n_codewords, n_jobs=-1, verbose=True, transform_algorithm='lasso_lars')
	skels_dict.fit(skels)

	feature_dictionaries = {'hog':hog_dict, 'hof':hof_dict, 'skel':skels_dict}

	with open('MSR_Dictionaries_hog-hof-skel_%f.dat'%time.time(), 'wb') as outfile:
	    pickle.dump(feature_dictionaries, outfile, protocol=pickle.HIGHEST_PROTOCOL)

# def chi_squared_kernel(x, y):
# 	return 1 - np.sum((2*(x-y)**2)/(x+y))

def label2ind(labels):
	for i,l in zip(range(len(labels)), labels):
		labels[np.where(labels==l)] = i
	return labels.astype(np.int)

def create_bag_of_words(feature_filename, dictionary_filename):
	
	''' Load dictionaries '''
	#filename = 'MSR_Dictionaries_hog-hof-skel_1360425153.84-hof-skel1360423760.27.dat'
	feature_dictionaries = np.load(feature_filename)
	hog_dict = feature_dictionaries['hog']
	hof_dict = feature_dictionaries['hof']
	skel_dict = feature_dictionaries['skel']
	label_set = read_MSR_labels()

	''' Create BOW histograms '''
	# dictionary_filename = 'MSR_Features_hog-hof-skel1360423760.27.dat'
	dataset_features = np.load(dictionary_filename)
	hogs = []
	hofs = []
	skels = []
	labels = []
	for n in dataset_features.keys():
		hogs += [np.abs(np.mean(hog_dict.transform(np.array(dataset_features[n]['hog'])), 0))]
		hofs += [np.abs(np.mean(hof_dict.transform(np.array(dataset_features[n]['hof']))))]
		skels += [np.abs(np.mean(skel_dict.transform(normalize_skeleton(np.array(dataset_features[n]['skel_world'])))))]
		labels += [label_set[int(n[1:3])-1]]

	hogs = np.vstack(hogsH)
	hofs = np.vstack(hofs)
	skels = np.vstack(skels)
	labels = np.array(labels)
	label_inds = label2ind(labels)

	BOW = {'hog':hogs, 'hof':hofs, 'skel':skels, 'labels':labels, 'label_indices':label_inds}

	with open('MSR_BOW_hog-hof-skel_%f.dat'%time.time(), 'wb') as outfile:
	    pickle.dump(BOW, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def cross_validate_bow(filename):
	'''
	Adapted from this example: http://scikit-learn.org/stable/auto_examples/grid_search_digits.html#example-grid-search-digits-py
	'''
	from sklearn.cross_validation import train_test_split
	from sklearn.grid_search import GridSearchCV
	from sklearn.metrics import classification_report
	from sklearn.metrics import precision_score
	from sklearn.metrics import recall_score
	from sklearn.svm import SVC
	from sklearn.kernel_approximation import AdditiveChi2Sampler

	chi = AdditiveChi2Sampler()
	chi.fit(hogsH, labels)
	X = chi.fit_transform(hogsH, labels)

	# clf = svm.SVC(kernel='rbf', C=100)
	# clf.fit(X, np.array(labels))
	# print "Training accuracy: %f"%(clf.score(X, labels)*100.)

	scores = [('precision', precision_score),
	    	('recall', recall_score),]
	for score_name, score_func in scores:

		X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=0)
		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
		                	{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

		clf = GridSearchCV(SVC(C=1), tuned_parameters, score_func=score_func)
		clf.fit(X_train, y_train, cv=5)

		print "Best parameters set found on development set:"
		print
		print clf.best_estimator_
		print
		print "Grid scores on development set:"
		print
		for params, mean_score, scores in clf.grid_scores_:
		    print "%0.3f (+/-%0.03f) for %r" % (
		        mean_score, scores.std() / 2, params)
		print

		print "Detailed classification report:"
		print
		print "The model is trained on the full development set."
		print "The scores are computed on the full evaluation set."
		print
		y_true, y_pred = y_test, clf.predict(X_test)
		print classification_report(y_true, y_pred)
		print
		print "Best score: %f"%clf.best_score_

'''
-----------------------------------
--------------MAIN ----------------
-----------------------------------
'''

if __name__=="__main__":
	if 1:
		main_calculate_features()
	elif 0:
		create_dictionaries()
	else:
		create_back_of_words()



