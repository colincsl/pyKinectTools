"""
Main file for training multi-camera pose
"""

#import os
import itertools as it
import optparse
#import time
import cPickle as pickle
import numpy as np
import cv2
import scipy.misc as sm
import scipy.ndimage as nd
from skimage import color
import skimage
from skimage.color import rgb2gray,gray2rgb
from skimage.feature import hog, local_binary_pattern, match_template, peak_local_max
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from pyKinectTools.utils.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.utils.DepthUtils import world2depth, depthIm2XYZ, skel2depth, depth2world
from pyKinectTools.utils.SkeletonUtils import display_skeletons, transform_skels, kinect_to_msr_skel
from pyKinectTools.algs.GeodesicSkeleton import *#generateKeypoints, distance_map
from pyKinectTools.algs.RandomForestPose import RFPose
from pyKinectTools.algs.HistogramOfOpticalFlow import hog2image
from pyKinectTools.utils.VideoViewer import *

from sklearn.kernel_approximation import SkewedChi2Sampler, AdditiveChi2Sampler, RBFSampler
#from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
from sklearn.linear_model import SGDClassifier


""" Debugging """
from IPython import embed
np.seterr(all='ignore')

from joblib import Parallel, delayed


'''
Felzenszwalb w/ 1 channel 133 ms per loop
Felzenszwalb w/ 3 channels 420 ms per loop
Adaptive 55ms per loop
HOG on person bounding box: 24 ms
HOG on person whole body: 101 ms for 4x4 px/cell and 70 ms for 8x8 px/cell for 24*24 px boxes
HOG per extrema 2-3 ms

It's not computationally efficient to compute hogs everywhere? What about multi-threaded?
'''

class MultiChannelChi2:
	chi_kernels = []

	def fit(data):
		'''
		data : [set_a, set_b, set_c, ...]
		data should only be positive
		'''
		for i in len(data):
			self.chi_kernels += [AdditiveChi2Sampler()]
			self.chi_kernels[-1].fit(data[i])


	def transform(data):
		output = []
		for i in len(data):
			output += [self.chi_kernels[i].transform(data[i])]

	def fit_transform(data):
		pass

def recenter_image(im):
	'''
	'''
	n_height, n_width = im.shape
	com = nd.center_of_mass(im)
	if any(np.isnan(com)):
		return im
	
	im_center = im[(com[0]-n_height/2):(com[0]+n_height/2)]
	offset = [(n_height-im_center.shape[0]),(n_width-im_center.shape[1])]
	
	if offset[0]%2 > 0:
		h_odd = 1
	else:
		h_odd = 0
	if offset[1]%2 > 0:
		w_odd = 1
	else:
		w_odd = 0			
	
	im[offset[0]/2:n_height-offset[0]/2-h_odd, offset[1]/2:n_width-offset[1]/2-w_odd] = im_center

	return im



def train(all_joint_ims_c, all_joint_ims_z, all_joint_labels):
	save_data = {}
	ims_z = np.array(all_joint_ims_z)
	ims_c = np.array(all_joint_ims_c)
	labels = np.array(all_joint_labels)
	n_samples = labels.shape[0]
	n_height, n_width = ims_c[0].shape

	# Relabel to only look at certain parts
	head, shoulders, hands, feet = [0], [2,5], [4,7], [10,13]
	classes = [head]+[hands]#+[shoulders]#+[feet]
	n_classes = len(classes)
	# classes = [head]+[hands]#+[feet]
	# Group into classes
	for c in xrange(n_samples):
		this_class = n_classes
		for i in xrange(n_classes):
			if labels[c] in classes[i]:
				this_class = i
				break
		labels[c] = this_class
		# ims_c[c] = recenter_image(ims_c[c])

	# Calculate HOGs and LBPs
	print 'Starting feature calculations'
	hog_size = (4,4)
	hog_cells = (3,3)
	save_data['hog_size']=hog_size
	save_data['hog_cells']=hog_cells
	hogs_c = Parallel(n_jobs=-1)(delayed(hog)(im, 9, (4,4), (3,3), True, False) for im in ims_c )
	hogs_c,hogs_im_c = zip(*hogs_c)
	hogs_c = np.array(hogs_c)
	
	lbp_px = 16
	lbp_radius = 2
	lbps_tmp = np.array(Parallel(n_jobs=-1)(delayed(local_binary_pattern)(im, P=16, R=2, method='uniform') for im in ims_c ))
	lbps_c = np.array(Parallel(n_jobs=-1)(delayed(np.histogram)(lbp, normed=True, bins = 18, range=(0,18)) for lbp in lbps_tmp ))
	lbps_c = np.array([x[0] for x in lbps_c])

	lbps_tmp = np.array(Parallel(n_jobs=-1)(delayed(local_binary_pattern)(im, P=16, R=2, method='uniform') for im in ims_z ))
	lbps_z = np.array(Parallel(n_jobs=-1)(delayed(np.histogram)(lbp, normed=True, bins = 18, range=(0,18)) for lbp in lbps_tmp ))
	lbps_z = np.array([x[0] for x in lbps_z])

	# Get rid of background
	lbps_c[:,lbps_c.argmax(1)]=0
	lbps_c = (lbps_c.T/lbps_c.sum(1)).T
	lbps_c = np.nan_to_num(lbps_c)
	lbps_z[:,lbps_z.argmax(1)]=0
	lbps_z = (lbps_c.T/lbps_c.sum(1)).T
	lbps_z = np.nan_to_num(lbps_c)

	## Dimensionality reduction on hogs
	# pca_c = PCA(13)#, whiten=True)
	# pca_z = PCA(13*2, whiten=True)
	# pca_both = PCA(2*(lbp_px+2), whiten=True)
	# training_c = pca_c.fit_transform(training_c)
	# training_z = pca_z.fit_transform(training_z)
	# training_both = pca_both.fit_transform(train_both)
	
	## Concatenate training vectors
	training_c = np.hstack([np.array(hogs_c), lbps_c])
	# training_z = np.hstack([training_z, lbps_z])
	training_both = np.array(hogs_c)
	# training_both = np.array(np.reshape(hogs_im_c, [-1,32*32]))
	# training_both = lbps_z
	# training_both = lbps_c
	# training_both = np.hstack([training_c, lbps_z])
	# training_both = np.hstack([np.array(hogs_c), lbps_c, lbps_z])

	# These are all histograms, ensure they are strictly positive
	hogs_c[hogs_c<0] = 0
	training_c[training_c<0] = 0
	# training_z[training_z<0] = 0
	training_both[training_both<0] = 0

	# Transfrom to chi/rbf space
	# chi_kernel_c = AdditiveChi2Sampler()
	# chi_kernel_both = AdditiveChi2Sampler(sample_steps=1)
	# save_data['Kernel_color'] = chi_kernel_c
	# save_data['Kernel_both'] = chi_kernel_both
	# training_both_kernel = chi_kernel_both.fit_transform(training_both)
	# training_both_kernel = np.exp(-training_both_kernel / training_both_kernel.mean())
	training_both_kernel = training_both

	# chi_kernel_all = [AdditiveChi2Sampler(sample_steps=1)]*3
	# a = chi_kernel_all[0].fit_transform(np.array(hogs_c))
	# b = chi_kernel_all[1].fit_transform(lbps_c)
	# c = chi_kernel_all[2].fit_transform(lbps_z)

	# rbf_kernel_both = RBFSampler()
	# training_both_kernel = rbf_kernel_both.fit_transform(training_both_kernel)

	# Classify
	print 'Starting classification training'
	# svm_c = SGDClassifier(class_weight="auto")
	# save_data['SVM_c'] = svm_c	
	svm_both = SGDClassifier(n_iter=100, alpha=.0001, class_weight="auto", l1_ratio=0, fit_intercept=True, n_jobs=-1)
	# svm_both = LinearSVC()
	svm_both.fit(training_both_kernel, labels)
	print "Done fitting both SVM. Self score: {0:.2f}%".format(svm_both.score(training_both_kernel, labels)*100)
	save_data['SVM_both'] = svm_both

	filters = [hog2image(c, [ims_c[0].shape[0],ims_c[0].shape[1]], pixels_per_cell=hog_size, cells_per_block=[1,1]) for c in svm_both.coef_]
	save_data['filters'] = filters
	# filters = [f*(f>0) for f in filters]


	# rf = RandomForestClassifier(n_estimators=50)
	# rf.fit(training_both_kernel, labels)
	# print "Done fitting forest. Self score: {0:.2f}%".format(rf.score(training_both_kernel, labels)*100)
	# save_data['rf'] = rf

	# Grid search for paramater estimation
	if 0:
		from sklearn.grid_search import GridSearchCV
		from sklearn.cross_validation import cross_val_score
		params = {'alpha': [.0001],
				'l1_ratio':[0.,.25,.5,.75,1.]}
		grid_search = GridSearchCV(svm_both, param_grid=params, cv=2, verbose=3)	
		grid_search.fit(training_both_kernel, labels)


	# Save data
	save_data['Description'] = 'Data:2500f CIRL;'
	with open('PCA_SVM.dat', 'w') as f:
		pickle.dump(save_data, f)
	print 'Parameters saved. Process complete'




# # Visualize
if 0:
	for j in xrange(n_classes+1):
		figure(j)
		i=0
		i2=0
		while i < 25:
			i2 += 1
			if labels[i2] == j:
				subplot(5,5,i); 
				# imshow(ims_c[i2,:,:])
				imshow(hogs_im_c[i2])
				# imshow(lbps_tmp[i2])
				i += 1
	show()

	# Viz filters
	for i,im in enumerate(filters):
		ii = i*2
		subplot(3,2,ii+1)
		imshow(im*(im>0))
		subplot(3,2,ii+2)
		imshow(im*(im<0))	
	show()



# -------------------------MAIN------------------------------------------

def main(visualize=False, learn=False):
	# Init both cameras
	# fill = True
	fill = False
	get_color = True
	cam = KinectPlayer(base_dir='./', device=1, bg_subtraction=True, get_depth=True, get_color=get_color, get_skeleton=True, fill_images=fill)	
	cam2 = KinectPlayer(base_dir='./', device=2, bg_subtraction=True, get_depth=True, get_color=get_color, get_skeleton=True, fill_images=fill)
	# Transformation matrix from first to second camera
	data = pickle.load(open("Registration.dat", 'r'))
	transform_c1_to_c2 = data['transform']

	data = pickle.load(open('PCA_SVM.dat', 'r'))
	hog_size = data['hog_size']
	hog_cells = data['hog_cells']
	# svm_c = data['SVM_color']
	svm_both = data['SVM_both']	
	# kernel_c = data['Kernel_color']
	# kernel_both = data['Kernel_both']
	filters = data['filters']
	n_filters = len(filters)
	# rf = data['rf']

	vv = VideoViewer()

	p_size = 24
	frame_count = 0

	if learn:
		all_joint_ims_z = []
		all_joint_ims_c = []
		all_joint_labels = []
	else:
		true_pos = {'hands':0, 'head':0}
		false_pos = {'hands':0, 'head':0}
	
	# cam.next(500)
	while cam.next(1) and frame_count < 2500:
	# while cam.next(1) and frame_count < 1000:
	# while cam.next(20) and frame_count < 100:
		if get_color:
			cam.colorIm = cam.colorIm[:,:,[2,1,0]]
		gray_im = (rgb2gray(cam.colorIm)*255).astype(np.uint8)
		height, width = cam.depthIm.shape
		
		# Update frames
		# cam2.sync_cameras(cam)

		# Transform skels from cam1 to cam2 and get rid of bad skeletons
		cam_skels = [np.array(cam.users[s]['jointPositions'].values()) for s in cam.users.keys()]
		cam_skels = [s for s in cam_skels if np.all(s[0] != -1)]

		# Check for skeletons
		if len(cam_skels) == 0:
			continue
		frame_count+=1
		print frame_count		

		mask = cam.get_person()
		box = nd.find_objects(mask>0)[0]
		if mask is not -1:
			mask = mask > 0
		else:
			continue
		cam.depthIm *= mask
		gray_im *= mask
		# cam.colorIm *= mask[:,:,None]>0

		skel_tmp = skel2depth(cam_skels[0], [240,320])

		# Save images
		if learn:

			extrema = geodesic_extrema_MPI(cam.depthIm*(mask>0), iterations=3)

			# Find joint label closest to each extrema
			if 0:
				for i,_ in enumerate(extrema):
					# j_pos = skel_tmp[i]
					j_pos = extrema[i]
					x = j_pos[0]
					y = j_pos[1]
					if x-p_size >= 0 and x+p_size < cam.depthIm.shape[0] and y-p_size >= 0 and y+p_size < cam.depthIm.shape[1]:
						all_joint_ims_z += [gray_im[x-p_size:x+p_size, y-p_size:y+p_size].astype(np.uint8)]
						all_joint_ims_c += [(gray_im[x-p_size:x+p_size, y-p_size:y+p_size]/(5000/255.)).astype(np.uint8)]
						# joint_labels += [i]
						dists = np.sqrt(np.sum((j_pos - skel_tmp[:,[1,0]])**2,-1))
						# print np.min(dists)
						all_joint_labels += [np.argmin(dists)]


			# Use offsets surounding joints
			if 1:
				# print skel_tmp
				for i,_ in enumerate(skel_tmp):
					joint_ims_c = []
					joint_ims_z = []
					joint_labels = []

					j_pos = skel_tmp[i]
					offset_count = 1
					for j in xrange(offset_count):
						offset = [0,0]#np.random.uniform(offset_max, size=[2])
						x = j_pos[1]# + offset[0]
						y = j_pos[0]# + offset[1]
						if x-p_size-offset[0] >= 0 and x+p_size+offset[0] < cam.depthIm.shape[0] and y-p_size-offset[1] >= 0 and y+p_size+offset[1] < cam.depthIm.shape[1]:
							joint_ims_c += [gray_im[x-p_size:x+p_size, y-p_size:y+p_size].astype(np.uint8)]
							joint_ims_z += [(gray_im[x-p_size:x+p_size, y-p_size:y+p_size]/(5000/255.)).astype(np.uint8)]
							joint_labels += [i]
					if len(joint_ims_z) > 0:
						all_joint_ims_z += joint_ims_z
						all_joint_ims_c += joint_ims_c
						all_joint_labels += joint_labels

		else:
			# cam2_skels = transform_skels(cam_skels, transform_c1_to_c2, 'image')

			# Calculate HOGs in grid. Then predict
			if 1:
				# hog_size = [8,8]
				b_height,b_width = gray_im[box].shape
				if gray_im[box].shape[0] < filters[0].shape[0] or gray_im[box].shape[1] < filters[0].shape[1]:
					continue
				hogs_array_c, hogs_im_c = hog(gray_im[box], orientations=9, pixels_per_cell=hog_size, cells_per_block=hog_cells, visualise=True, normalise=False)
				embed()
				# SVM
				if 1:
					hog_layers = []
					for f in filters:
						hog_layers += [match_template(hogs_im_c, f, pad_input=True)]

					hog_layers = np.dstack(hog_layers)
					mask_tmp = np.maximum((hog_layers.max(-1) == 0), -mask[box])
					predict_class = hog_layers[0::8,0::8].argmax(-1)*(-mask_tmp[::8,::8]) + ((-mask_tmp[::8,::8])*1)
					predict_prob = hog_layers[0::8,0::8].max(-1)*(-mask_tmp[::8,::8]) + ((-mask_tmp[::8,::8])*1)

					output = np.zeros([predict_class.shape[0], predict_class.shape[1], n_filters], dtype=np.float)
					for i in range(n_filters):
						output[predict_class==i+1,i] = predict_class[predict_class==i+1]+predict_prob[predict_class==i+1]
						if 0 and i != 2:
							for ii,jj in it.product(range(8),range(8)):
								try:
									cam.colorIm[box][ii::8,jj::8][predict_class==i+1] = (255*(i==0),255*(i==1),255*(i==2))
								except:
									pass
				# Random Forest
				if 0:
					h_count = hogs_array_c.shape[0] / 9
					hog_height, hog_width = [b_height/8, b_width/8]
					square_size = p_size*2/8
					hogs_square = hogs_array_c.reshape(hog_height, hog_width, 9)
					# embed()
					predict_class = np.zeros([hog_height, hog_width])
					predict_prob = np.zeros([hog_height, hog_width])
					output = predict_class
					for i in range(0,hog_height):
						for j in range(0,hog_width):
							if i-square_size/2 >= 0 and i+square_size/2 < hog_height and j-square_size/2 >= 0 and j+square_size/2 < hog_width:
								predict_class[i,j] = rf.predict(hogs_square[i-square_size/2:i+square_size/2,j-square_size/2:j+square_size/2].flatten())[0]
								predict_prob[i,j] = rf.predict_proba(hogs_square[i-square_size/2:i+square_size/2,j-square_size/2:j+square_size/2].flatten()).max()
					
				n_peaks = [2,2,2]
				for i in range(n_filters):
					optima = peak_local_max(predict_prob*(predict_class==i+1), min_distance=2, num_peaks=n_peaks[i], exclude_border=False)
					for o in optima:
						joint = o*8 + [box[0].start, box[1].start]
						circle = skimage.draw.circle(joint[0],joint[1], 15)
						circle = np.array([np.minimum(circle[0], height-1),
											np.minimum(circle[1], width-1)])
						circle = np.array([np.maximum(circle[0], 0),
											np.maximum(circle[1], 0)])

						cam.colorIm[circle[0], circle[1]] = (255*(i==0),255*(i==1),255*(i==2))

				if 0:
					for i in range(n_filters-1):
						figure(i+1)
						imshow(predict_prob*(predict_class==i+1))
					show()

				# tmp = gray2rgb(sm.imresize(predict_class, np.array(predict_class.shape)*10, 'nearest'))
				# tmp[:,:,2] = 255 - tmp[:,:,2]
				tmp = sm.imresize(output, np.array(predict_class.shape)*10, 'nearest')
				cv2.imshow("O", tmp/float(tmp.max()))
				cv2.waitKey(10)


			# Calculate HOGs at geodesic extrema. The predict
			if 0:
				height, width = cam.depthIm.shape
				n_px = 16
				n_radius = 2

				extrema = geodesic_extrema_MPI(cam.depthIm*(mask>0), iterations=10)
				extrema = np.array([e for e in extrema if p_size<e[0]<height-p_size and p_size<e[1]<width-p_size])
				# joint_names = ['head', 'torso', 'l_shoulder', 'l_elbow', 'l_hand',\
				# 				'r_shoulder', 'r_elbow', 'r_hand',\
				# 				'l_hip', 'l_knee', 'l_foot', \
				# 				'r_hip', 'r_knee', 'r_foot']
				joint_names = ['head', 'l_hand', 'l_foot', 'other']						
				
				# Depth
				if 1:
					# hogs_z = [hog(cam.depthIm[e[0]-p_size:e[0]+p_size, e[1]-p_size:e[1]+p_size], pixels_per_cell=[8,8], , False, True) for e in extrema]
					# hogs_z_pca = np.array([pca_z.transform(hogs_z[i]) for i in range(len(hogs_z))]).reshape([len(extrema), -1])
					# hogs_z_pca = hogs_z
					lbp_tmp = [local_binary_pattern(cam.depthIm[e[0]-p_size:e[0]+p_size, e[1]-p_size:e[1]+p_size], P=n_px, R=n_radius, method='uniform') for e in extrema]
					lbp_hists_z = np.array([np.histogram(im, normed=True, bins = n_px+2, range=(0,n_px+2))[0] for im in lbp_tmp])
					lbp_hists_z[lbp_hists_z.argmax(0)] = 0
					lbp_hists_z = lbp_hists_z.T / lbp_hists_z.max(1)
					lbp_hists_z = np.nan_to_num(lbp_hists_z)			
					# hogs_z_pca = np.hstack([hogs_z_pca, lbp_hists_z])
					# hogs_z_pred = np.hstack([svm_z.predict(h) for h in hogs_z_pca]).astype(np.int)
					# names = [joint_names[i] for i in hogs_z_pred]

				# Color
				if 1:
					hogs_c = [hog(gray_im[e[0]-p_size:e[0]+p_size, e[1]-p_size:e[1]+p_size], 9, (8,8), (3,3), False, True) for e in extrema]					
					# hogs_c = [hog(gray_im[e[0]-p_size:e[0]+p_size, e[1]-p_size:e[1]+p_size], 9, (8,8), (3,3), True, True) for e in extrema]
					# hogs_c, hogs_im_c = zip(*hogs_c)
					# hogs_c_pca = np.array([pca_c.transform(hogs_c[i]) for i in range(len(hogs_c))]).reshape([len(extrema), -1])
					hogs_c_pca = hogs_c

					# lbp_tmp = np.array(Parallel(n_jobs=-1)(delayed(local_binary_pattern)(im, 16, 2, 'uniform') for im in ims ))
					lbp_tmp = [local_binary_pattern(gray_im[e[0]-p_size:e[0]+p_size, e[1]-p_size:e[1]+p_size], P=n_px, R=n_radius, method='uniform') for e in extrema]
					lbp_hists_c = np.array([np.histogram(im, normed=True, bins = n_px+2, range=(0,n_px+2))[0] for im in lbp_tmp])
					lbp_hists_c[lbp_hists_c.argmax(0)] = 0
					lbp_hists_c = lbp_hists_c.T / lbp_hists_c.max(1)
					lbp_hists_c = np.nan_to_num(lbp_hists_c)
					# embed()
					# hogs_c_pca = np.hstack([hogs_c_pca, lbp_hists_c])
					# hogs_c_pred = np.hstack([svm_c.predict(h) for h in hogs_c_pca]).astype(np.int)
					# names = [joint_names[i] for i in hogs_c_pred]
				# embed()
				# Both
				if 1:
					# data_both_pca = np.array([pca_both.transform(data_both[i]) for i in range(len(data_both))]).reshape([len(extrema), -1])
					# data_both_pca = data_both
					# embed()
					data_both_pca = np.hstack([np.array(hogs_c), lbp_hists_c.T, lbp_hists_z.T])
					data_both_pca[data_both_pca<0] = 0
					# data_both_pca = kernel_both.transform(data_both_pca)
					# data_both_pca = np.exp(data_both_pca)
					
					hogs_both_pred = np.hstack([svm_both.predict(h) for h in data_both_pca]).astype(np.int)
					names = [joint_names[i] for i in hogs_both_pred]
					skel_predict = hogs_both_pred.astype(np.int)				

				# print names
				im_c = cam.colorIm
				d = p_size
				for i in range(len(extrema)):
					color = 0
					if names[i] == 'head':
						color = [255,0,0]
					elif names[i] in ('l_hand', 'r_hand'):
						color = [0,255,0]
					# elif names[i] == 'l_shoulder' or names[i] == 'r_shoulder':
						# color = [0,0,255]
					# elif names[i] == 'l_foot' or names[i] == 'r_foot':
						# color = [0,255,255]							
					# else:
						# color = [0,0,0]

					if color != 0:
						im_c[extrema[i][0]-d:extrema[i][0]+d, extrema[i][1]-d:extrema[i][1]+d] = color
					else:
						im_c[extrema[i][0]-d/2:extrema[i][0]+d/2, extrema[i][1]-d/2:extrema[i][1]+d/2] = 0
					# im_c[extrema[i][0]-d:extrema[i][0]+d, extrema[i][1]-d:extrema[i][1]+d] = lbp_tmp[i][:,:,None] * (255/18.)
						# cv2.putText(im_c, names[i], (extrema[i][1], extrema[i][0]), 0, .4, (255,0,0))
				# cv2.imshow("Label_C", im_c*mask[:,:,None])
				# cv2.waitKey(10)

				# Accuracy
				print skel_predict
				extrema_truth = np.empty(len(extrema), dtype=np.int)
				for i in range(len(extrema)):
					ex = extrema[i]
					dist = np.sqrt(np.sum((ex - skel_tmp[:,[1,0]])**2,-1))
					extrema_truth[i] = np.argmin(dist)

					# print skel_predict[i], extrema_truth[i]
					if skel_predict[i] == 1:
						# if extrema_truth[i] == 4 or extrema_truth[i] == 7:
						if extrema_truth[i] in (11, 7):
							true_pos['hands'] += .5
						else:
							false_pos['hands'] += .5
					elif skel_predict[i] == 0:
						if extrema_truth[i] == 3:
							true_pos['head'] += 1
							print 'h correct'
						else:
							false_pos['head'] += 1
				print "Hands", true_pos['hands'] / float(frame_count)#float(false_pos['hands'])
				print "Head", true_pos['head'] / float(frame_count)##float(false_pos['head'])
				# embed()

			if 1:#visualize:
				# cam2.depthIm = display_skeletons(cam2.depthIm, cam2_skels[0], (5000,), skel_type='Low')
				# cam.depthIm = display_skeletons(cam.depthIm, skel_tmp, (5000,), skel_type='Low')
				skel_tmp = kinect_to_msr_skel(skel_tmp)
				cam.colorIm[:,:,2] = display_skeletons(cam.colorIm[:,:,2], skel_tmp, (255,), skel_type='Low')
				cam.visualize()
				# cam2.visualize()

	if learn:
		# train(all_joint_ims_c, all_joint_ims_z, all_joint_labels)
		embed()

	print 'Done'

def HumanHogs(im):
	'''
	im should just contain the person (the bounding box)
	'''
	from skimage.feature import hog
	if im.dtype != np.uint8:
		im = (im/float(im.max()/255.)).astype(np.uint8)
	
	height, width = im.shape
	hogs = []
	for u in xrange(0, height, 24):
		for v in xrange(0, width, 24):
			if u+24 < height-1 and v+24 < width-1:
				im_tmp = im[u:u+24, v:v+24]
				hArray = hog(im_tmp, pixels_per_cell=[4,4])
				hogs += [hArray]

	return hogs

if 0:


	figure(4)
	names = [joint_names[i] for i in hogs_z_pred.astype(np.int)]
	labels_resized = sm.imresize(hogs_z_pred.reshape([-1, 5]), im.shape, 'nearest')
	matshow(labels_resized/13/10. + im/float(im.max()))
	im_c = cam.colorIm[box[0]]
	# matshow(labels_resized/13/10. + im/float(im.max()))
	matshow(labels_resized/13/10. + im_c[:,:,0]/float(im_c.max()))


if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	parser.add_option('-l', '--learn', dest='learn', action="store_true", default=False, help='Training phase')
	# parser.add_option('-i', '--infer', dest='infer', action="store_true", default=False, help='Training phase')
	(opt, args) = parser.parse_args()

	main(visualize=opt.viz, learn=opt.learn)
