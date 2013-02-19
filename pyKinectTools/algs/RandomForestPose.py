'''
Random Forest-based human pose estimation
Author: Colin Lea

References:
Conference paper: [CVPR]
Journal Paper: [PAMI]

-Learning-
For each training image calculate 500 features per pixel location and run a decision forest. 
The class of each pixel depends on the closest labeled joint location as determined by the geodesic distances

-Inference-
For each testing image run the forest for every pixel location.
Do mean shift on each cluster.

'''

'''
TODO: add constaint to skeletons so they lie within the silhouette 
'''

import os, time, pickle
import numpy as np
import scipy.misc as sm
import cv2
# from skimage.segmentation import quickshift
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from pyKinectTools.utils.SkeletonUtils import display_MSR_skeletons
from pyKinectTools.utils.DepthUtils import world2depth
from pyKinectTools.dataset_readers.MSR_DailyActivities import read_MSR_depth_ims, read_MSR_color_ims, read_MSR_skeletons, read_MSR_labels
from pyKinectTools.algs.BackgroundSubtraction import extract_people
from pyKinectTools.algs.GeodesicSkeleton import generateKeypoints, distance_map
from pyKinectTools.utils.VideoViewer import VideoViewer

from IPython import embed
# vv = VideoViewer()



''' ---------------- Learning ---------------- '''

def create_rf_offsets(offset_max=250, feature_count=500, seed=0):
	'''
	Defaults are the max offset variability and feature count shown in [PAMI].

	From [PAMI] Sec 3.1, the second offsets list is 0 with probability 0.5
	'''
	np.random.seed(seed=seed)
	offsets_1 = (np.random.rand(feature_count,2)-0.5)*offset_max
	offsets_2 = (np.random.rand(feature_count,2)-0.5)*offset_max
	offsets_2[np.random.random(offsets_2.shape[0]) < 0.5] = 0

	return offsets_1, offsets_2


def calculate_rf_features(im, offsets_1, offsets_2):
	'''
	im : masked depth image
	'''
	# Get u,v positions for each pixel location
	pixels = np.nonzero(im > 0)
	px_count = pixels[0].shape[0]

	# Get depths of each pixel location
	# Convert 8bit value to real value with ratio 8000/255
	depths = im[pixels] * 8000./255
	pixels = np.array(pixels).T
	n_features = len(offsets_1)

	output = np.zeros([px_count, n_features], dtype=np.int16)
	height, width = im.shape

	''' 
	For each index get the feature offsets
			f(u) = depth(u + offset_1/depth(u)) - depth(u + offset_2/depth(u))
	'''
	for i in xrange(n_features):
		# Find offsets for whole image
		dx = pixels + offsets_1[i]/(depths[:,None]/1000.)
		dy = pixels + offsets_2[i]/(depths[:,None]/1000.)

		# Ensure offsets are within bounds
		in_bounds_x = (dx[:,0]>=0)*(dx[:,0]<height)*(dx[:,1]>=0)*(dx[:,1]<width)
		out_of_bounds_x = True - in_bounds_x
		dx[out_of_bounds_x] = [0,0]
		dx = dx.astype(np.int)

		in_bounds_y = (dy[:,0]>=0)*(dy[:,0]<height)*(dy[:,1]>=0)*(dy[:,1]<width)
		out_of_bounds_y = True - in_bounds_y
		dy[out_of_bounds_y] = [0,0]
		dy = dx.astype(np.int)

		# Calculate actual offsets
		diffx = im[dx[:,0],dx[:,1]]
		diffy = im[dy[:,0],dy[:,1]]

		diff = diffx-diffy
		diff[out_of_bounds_y] = 20000
		diff[out_of_bounds_x] = 20000
		output[:,i] = diff

	return output

def get_per_pixel_joints(im_depth, skel_pos):
	'''
	Find the closest joint to each pixel using geodesic distances.

	im_depth : should be masked depth image
	skel_pos : 
	'''
	height, width = im_depth.shape
	distance_ims = []
	# Only look at major joints
	# for pos in skel_pos:
	for i in [0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17, 19]:	
		pos = skel_pos[i]
		''' It's a little faster if you only look at the body -- but segfaults?'''
		# _, bounding_box, n_boxes = extract_people(im_depth, im_depth > 0)
		# bounding_box = bounding_box[0]
		# corner = bounding_box[0]
		# bb_dims = [bounding_box[0].stop-bounding_box[0].start, bounding_box[1].stop-bounding_box[1].start]
		# x = np.maximum(np.minimum(pos[1]-bounding_box[0].start, bb_dims[0]-1), 0)
		# y = np.maximum(np.minimum(pos[0]-bounding_box[1].start, bb_dims[1]-1), 0)		
		# im_dist = distance_map(im_depth[bounding_box], centroid=[x,y])

		''' Without bounding box '''
		x = np.maximum(np.minimum(pos[1], height-1), 0)
		y = np.maximum(np.minimum(pos[0], width-1), 0)
		im_dist = distance_map(im_depth, centroid=[x,y])
		
		distance_ims += [im_dist]

	closest_pos = np.argmin(distance_ims, 0).astype(np.uint8)
	# closest_pos[im_depth[bounding_box]==0] = 100

	return closest_pos

''' Unncessesary? '''
def learn_rf(features, labels):
	'''
	'''
	n_estimators = 10
	criterion = 'entropy'
	max_depth = 18
	max_features='auto'
	n_jobs=-1
	compute_importances=True
	oob_score=True
	random_state=None
	verbose=0

	rf = RFClassifier(n_estimators=n_estimators, criterion=criterion,\
	 					max_depth=max_depth, max_features=max_features,\
	  					compute_importances=compute_importances, oob_score=oob_score,\
	  					n_jobs=n_jobs, random_state=random_state, verbose=verbose,\
	  					min_samples_leaf=1)	

	#rf.fit(features, labels)

	return rf


# def frame_features():


def main_learn():
	'''
	'''

	offsets_1, offsets_2 = create_rf_offsets()

	name = 'a01_s02_e02_'
	depth_file = name + "depth.bin"
	color_file = name + "rgb.avi"
	skeleton_file = name + "skeleton.txt"
	''' Read data from each video/sequence '''
	try:
		depthIms, maskIms = read_MSR_depth_ims(depth_file)
		colorIms = read_MSR_color_ims(color_file)
		skels_world, skels_im = read_MSR_skeletons(skeleton_file)
	except:
		print "Error reading data"


	all_features = []
	all_labels = []
	for i in xrange(len(depthIms)):
		try:
		# if 1:
			print i
			''' Get frame data '''
			im_depth = depthIms[i]
			im_color = colorIms[i]
			im_mask = maskIms[i]
			skel_pos = world2depth(skels_world[i], rez=[240,320])

			''' Compute features and labels '''
			im_depth *= im_mask
			features = calculate_rf_features(im_depth, offsets_1, offsets_2)
			im_labels = get_per_pixel_joints(im_depth, skel_pos)
			# print i, 3
			im_labels[im_depth==0] = 25			
			pixel_labels = im_labels[np.nonzero(im_labels < 25)]

			all_features += [features]
			all_labels += [pixel_labels]
			im_labels = np.repeat(im_labels[:,:,None], 3, -1)
			im_labels = display_MSR_skeletons(im_labels, skel_pos, (20,))
			# cv2.imshow("feature_space", im_labels/im_labels.max().astype(np.float))
			# cv2.imshow("color", im_color)
			# ret = cv2.waitKey(10)
			# if ret > 0: break

		except:
			print "Frame failed:", i

	embed()

	all_features2 = np.vstack(all_features)
	all_labels2 = np.hstack(all_labels)

	# data = {'features':all_features2, 'labels':all_labels2}
	# with open("0_2_2_features.dat", 'w') as f:
	# 	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

	rf.fit(all_features2, all_labels2)

	# Save forest
	with open("0_2_2_RF.dat", 'w') as f:
		pickle.dump(rf, f, pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
	main_learn()


if 0:
	offsets_1, offsets_2 = create_rf_offsets()

	name = 'a01_s02_e02_'
	depth_file = name + "depth.bin"
	color_file = name + "rgb.avi"
	skeleton_file = name + "skeleton.txt"
	''' Read data from each video/sequence '''
	try:
		depthIms, maskIms = read_MSR_depth_ims(depth_file)
		depthIms *= maskIms
		colorIms = read_MSR_color_ims(color_file)
		skels_world, skels_im = read_MSR_skeletons(skeleton_file)
	except:
		print "Error reading data"


	all_features = []
	all_labels = []
	i=0
	print i
	''' Get frame data '''
	im_depth = depthIms[i]
	im_color = colorIms[i]
	# im_mask = maskIms[i]
	skel_pos = world2depth(skels_world[i], rez=[240,320])

	''' Compute features and labels '''
	# im_depth *= im_mask
	features = calculate_rf_features(im_depth, offsets_1, offsets_2)

	im_labels = get_per_pixel_joints(im_depth, skel_pos)

	pixel_labels = im_labels[np.nonzero(im_depth>0)]

	all_features += [features]
	all_labels += [pixel_labels]



''' ---------------- Inference ---------------- '''






''' Parallelism '''
# from joblib import Parallel, delayed
# print "Computing with multiple threads"
# data = Parallel(n_jobs=-1)( delayed(compute_features)(n) for n in base_names )
# for n,i in zip(base_names, range(len(base_names))):
# 	if data[i] != -1:
# 		dataset_features[n] = data[i]



# i=0
# height, width = [240,320]
# for i in range(len(depthIms)):
# 	im_d = depthIms[i]
# 	im_c = colorIms[i]
# 	im_m = maskIms[i]
# 	im_s = world2depth(skels_world[i], rez=[240,320])
# 	distance_ims = []
# 	for pos in im_s:
# 		x = np.maximum(np.minimum(pos[1], height-1), 0)
# 		y = np.maximum(np.minimum(pos[0], width-1), 0)
# 		extrema, all_trails, im_dist = generateKeypoints(im_d*im_m, centroid=[x,y])
# 		im_dist[im_dist==0] = 32000
# 		distance_ims += [im_dist]
# 	closest_pos = np.argmin(distance_ims, 0)

# 	print i
# 	try:
# 		# im_out = display_MSR_skeletons(closest_pos, im_s)
# 		cv2.imshow("D", closest_pos/closest_pos.max().astype(np.float))
# 		# cv2.imshow("D", im_out/im_out.max().astype(np.float))
# 	except:
# 		print 'err'
# 		pass

# 	# im_out = display_MSR_skeletons(im_d, im_s)
# 	# cv2.imshow("D", im_out/im_out.max().astype(np.float))

# 	ret = cv2.waitKey(10)



