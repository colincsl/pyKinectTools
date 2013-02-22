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
Projection to world space (3.2 pg 8 [PAMI])

Feature calculations (w/ 60 features) runs at 6 fps

'''

import os, time, pickle
import numpy as np
import scipy.misc as sm
import cv2
# from skimage.segmentation import quickshift
from sklearn.ensemble import RandomForestClassifier as RFClassifier
#from sklearn.ensemble import ExtraTreesClassifier as RFClassifier
from skimage import draw
from pyKinectTools.utils.SkeletonUtils import display_MSR_skeletons
from pyKinectTools.utils.DepthUtils import world2depth
from pyKinectTools.dataset_readers.MSR_DailyActivities import read_MSR_depth_ims, read_MSR_color_ims, read_MSR_skeletons, read_MSR_labels
from pyKinectTools.algs.BackgroundSubtraction import extract_people
from pyKinectTools.algs.MeanShift import mean_shift
from pyKinectTools.algs.GeodesicSkeleton import generateKeypoints, distance_map
from pyKinectTools.utils.VideoViewer import VideoViewer

from IPython import embed
# from pylab import *
# vv = VideoViewer()


N_MSR_JOINTS = 20
MSR_JOINTS = range(N_MSR_JOINTS)
# SKEL_JOINTS = [0, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19]
SKEL_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19]
# SKEL_JOINTS = [0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17, 19]
N_SKEL_JOINTS = len(SKEL_JOINTS)


''' ---------------- Learning ---------------- '''

def create_rf_offsets(offset_max=250, feature_count=100, seed=0):
	'''
	Defaults are the max offset variability and feature count shown in [PAMI].
	feature_count should be 500

	From [PAMI] Sec 3.1, the second offsets list is 0 with probability 0.5
	'''
	np.random.seed(seed=seed)
	offsets_1 = (np.random.rand(feature_count,2)-0.5)*offset_max
	offsets_2 = (np.random.rand(feature_count,2)-0.5)*offset_max
	offsets_2[np.random.random(offsets_2.shape[0]) < 0.5] = 0

	return offsets_1, offsets_2


def calculate_rf_features(im, offsets_1, offsets_2, mask=None):
	'''
	im : masked depth image
	'''

	if mask is None:
		mask = 1

	# Get u,v positions for each pixel location
	pixels = np.nonzero((im*mask) > 0)
	px_count = pixels[0].shape[0]

	# Get depths of each pixel location
	# Convert 8bit value to real value with ratio 8000/255
	# depths = im[pixels] * 8000./255
	depths = im[pixels]
	pixels = np.array(pixels).T
	n_features = len(offsets_1)

	output = np.zeros([px_count, n_features], dtype=np.int16)
	height, width = im.shape

	''' 
	For each index get the feature offsets
			f(u) = depth(u + offset_1/depth(u)) - depth(u + offset_2/depth(u))
	'''
	im[im==0] = 20000
	for i in xrange(n_features):
		# Find offsets for whole image
		dx = pixels + offsets_1[i]/(depths[:,None]/1000.)
		dy = pixels + offsets_2[i]/(depths[:,None]/1000.)

		# Ensure offsets are within bounds
		in_bounds_x = (dx[:,0]>=0)*(dx[:,0]<height)*(dx[:,1]>=0)*(dx[:,1]<width)
		out_of_bounds_x = True - in_bounds_x
		dx[out_of_bounds_x] = [0,0]
		dx = dx.astype(np.int16)

		in_bounds_y = (dy[:,0]>=0)*(dy[:,0]<height)*(dy[:,1]>=0)*(dy[:,1]<width)
		out_of_bounds_y = True - in_bounds_y
		dy[out_of_bounds_y] = [0,0]
		dy = dy.astype(np.int16)

		# Calculate actual offsets
		diffx = im[dx[:,0],dx[:,1]]
		diffy = im[dy[:,0],dy[:,1]]

		diff = diffx-diffy
		diff[out_of_bounds_y] = 20000
		diff[out_of_bounds_x] = 20000
		output[:,i] = diff

	im[im==20000] = 0
	return output

def get_per_pixel_joints(im_depth, skel_pos, alg='geodesic', radius=5):
	'''
	Find the closest joint to each pixel using geodesic distances.
	---Paramaters---
	im_depth : should be masked depth image
	skel_pos : 
	alg : 'geodesic' or 'circular'
	radius : [only for circular]
	'''
	height, width = im_depth.shape
	distance_ims = []
	distance_ims = np.empty([height, width, N_SKEL_JOINTS])
	# Only look at major joints
	for i, j in zip(SKEL_JOINTS, range(N_SKEL_JOINTS)):
		pos = skel_pos[i]

		if alg == 'geodesic':
			# Get bounding box -- this is faster than whole image
			# _, bounding_box, n_boxes = extract_people(im_depth, im_depth > 0)
			# bounding_box = bounding_box[0]
			# corner = bounding_box[0]
			# bb_dims = [bounding_box[0].stop-bounding_box[0].start, bounding_box[1].stop-bounding_box[1].start]
			# Get distance map
			# x = np.maximum(np.minimum(pos[1]-bounding_box[0].start, bb_dims[0]-1), 0)
			# y = np.maximum(np.minimum(pos[0]-bounding_box[1].start, bb_dims[1]-1), 0)		
			# print 'a'
			# im_dist = distance_map(im_depth[bounding_box], centroid=[x,y]).copy()

			x = np.maximum(np.minimum(pos[1], height-1), 0)
			y = np.maximum(np.minimum(pos[0], width-1), 0)		
			im_dist = distance_map(im_depth, centroid=[x,y], scale=30.0)

			distance_ims[:,:,j] = im_dist.copy()
			# print 'c'

			if 0:
				from skimage.draw import circle
				radius = 5
				x = np.maximum(np.minimum(pos[1], height-1-radius), radius)
				y = np.maximum(np.minimum(pos[0], width-1-radius), radius)
				# print x,y
				pts = circle(x,y, radius)
				# print pts[0]
				max_ = distance_ims[distance_ims[:,:,j]!=32000, j].max()
				distance_ims[distance_ims[:,:,j]==32000,j] = max_+100
				distance_ims[pts[0],pts[1],j] = max_+100
				figure(j)
				imshow(distance_ims[:,:,j])

		elif alg == 'circular':
			x = np.maximum(np.minimum(pos[1], height-1-radius), radius)
			y = np.maximum(np.minimum(pos[0], width-1-radius), radius)
			pts = draw.circle(x,y, radius)
			closest_pos[pts] = i

		else:
			print "No algorithm type in 'get_per_pixel_joints'"


	if alg == 'geodesic':
		closest_pos = np.argmin(distance_ims, -1).astype(np.uint8)
		# closest_pos[bounding_box] = np.argmin(distance_ims, 0).astype(np.uint8)
	elif alg == 'circular':
		closest_pos = np.zeros_like(im_depth)
	
	#Change all background pixels
	closest_pos[im_depth==0] = N_SKEL_JOINTS+1
	# figure(100)
	# imshow(closest_pos)
	# embed()
	return closest_pos



def create_MSR_filenames(actions, subjects, positions):
	'''
	---Parameters---
	actions : list of numbers 1-16
	subjects: list of numbers 1-10
	positions: list of numbers netween 1-2 (1=sitting, 2=standing)

	---Returns---
	list of filenames
	'''

	import itertools as it
	filenames = []
	indicies = [i for i in it.product(actions, subjects, positions)]
	for i in indicies:
		filenames += ["a{0:02d}_s{1:02d}_e{2:02d}_".format(i[0],i[1],i[2])]
	
	return filenames


def learn_frame(name, offsets_1, offsets_2):
	'''
	'''
	depth_file = name + "depth.bin"
	color_file = name + "rgb.avi"
	skeleton_file = name + "skeleton.txt"
	''' Read data from each video/sequence '''
	try:
		depthIms, maskIms = read_MSR_depth_ims(depth_file)
		depthIms *= maskIms
		colorIms = read_MSR_color_ims(color_file)
		skels_world, _ = read_MSR_skeletons(skeleton_file)
	except:
		print "Error getting frame features"
		return -1,-1

	all_features = []
	all_labels = []
	for i in xrange(0, len(depthIms), 25):
		try:
			''' Get frame data '''
			im_depth = depthIms[i]
			skel_pos = world2depth(skels_world[i], rez=[240,320])
			''' --- NOTE THIS 10 PX OFFSET IN THE MSR DATASET !!! --- '''
			skel_pos[:,0] -= 10
			
			''' Compute features and labels '''
			im_labels = get_per_pixel_joints(im_depth, skel_pos, 'geodesic')
			im_labels[(im_depth>0)*(im_labels==255)] = 25
			mask = (im_labels<N_SKEL_JOINTS+1)
			pixel_loc = np.nonzero(mask)
			pixel_labels = im_labels[pixel_loc]	
			features = calculate_rf_features(im_depth, offsets_1, offsets_2, mask=mask)

			#print i, len(pixel_loc[0]), "Pixels"
			all_features += [features]
			all_labels += [pixel_labels]

			''' Visualize '''
			if 0:
				im_labels = np.repeat(im_labels[:,:,None], 3, -1)
				im_labels = display_MSR_skeletons(im_labels, skel_pos, (20,), skel_type='Low')
				cv2.putText(im_labels, "Blue=Truth", (10, 210), cv2.FONT_HERSHEY_DUPLEX, .5, (int(im_labels.max()/2), 0, 0))
				cv2.putText(im_labels, "Green=Predict", (10, 230), cv2.FONT_HERSHEY_DUPLEX, .5, (0, int(im_labels.max()/2), 0))					
				cv2.imshow("feature_space", im_labels/im_labels.max().astype(np.float))
				ret = cv2.waitKey(10)
				if ret > 0: break

		except:
			print "Error computing frame feature"
			pass

	del depthIms, maskIms, colorIms, skels_world
	print name, "Done"
	return np.concatenate(all_features), np.concatenate(all_labels)

# There is in issue extracting all the features at once
def chunks(l, n):
    ''' Yield successive n-sized chunks from l. '''
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def main_learn():
	'''
	'''

	offsets_1, offsets_2 = create_rf_offsets()
	all_features = []
	all_labels = []

	names = create_MSR_filenames(np.arange(5)+1, np.arange(1)+1, [2])

	# Parallelize feature collection
	try:
		from joblib import Parallel, delayed
		for n_set in chunks(names, 5):
			print "Computing with multiple threads. Current feature count:", len(all_features)
			'''
			all_data = learn_frame(n_set[0], offsets_1, offsets_2)
			#embed()
			if all_features == []:
				all_features = all_data[0]
				all_labels = all_data[1]
			else:
				all_features = np.vstack([all_features, all_data[0]])
				all_labels = np.hstack([all_labels, all_data[1]])
			'''

			all_data = Parallel(n_jobs=-1, verbose=True, pre_dispatch=2)( delayed(learn_frame)(n, offsets_1, offsets_2) for n in n_set )
			#embed()
			all_features += [f[0] for f in all_data if f[0] is not -1]
			all_labels += [f[1] for f in all_data if f[1] is not -1]

			print "Done computing this set of features/labels"

		print "Done computing all features/labels"
	except:
		print "Error computing features"
		return

	#embed()
	all_features = np.vstack(all_features)
	all_labels = np.hstack(all_labels)
	print "Starting forest"
	rf = RFClassifier(n_estimators=3,
						criterion='entropy',\
	 					max_depth=20, 
	 					max_features='auto',\
	  					oob_score=False,\
	  					n_jobs=-1, 
	  					random_state=None, 
	  					verbose=1,\
	  					min_samples_leaf=1)

	rf.fit(all_features, all_labels)

	save_data("../Saved_Params/"+str(time.time())+"_forests.dat", {'rf':rf, 'offsets':[offsets_1, offsets_2]})

	# embed()



def save_data(name, data):
	with open(name, 'w') as f:
	 	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


''' ---------------- Inference ---------------- '''


def main_infer(rf_name=None):
	'''
	'''

	if rf_name is None:
		import os
		files = os.listdir('../Saved_Params/')
		rf_name = '../Saved_Params/' + files[-1]
		print "Classifier file:",rf_name


	# Load classifier data
	data =  pickle.load(open(rf_name))
	rf = data['rf']
	offsets_1, offsets_2 = data['offsets']

	name = 'a01_s02_e02_'
	# name = 'a01_s07_e02_'
	# name = 'a02_s01_e02_'
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


	# all_pred = []
	all_pred_ims = []
	# for i in xrange(len(depthIms)):
	for i in xrange(2, len(depthIms), 3):
		# try:
		if 1:
			print i
			''' Get frame data '''
			im_depth = depthIms[i]
			skel_pos = world2depth(skels_world[i], rez=[240,320])
			''' --- NOTE THIS 10 PX OFFSET IN THE MSR DATASET !!! --- '''
			skel_pos[:,0] -= 10

			''' Compute features and labels '''
			features = calculate_rf_features(im_depth, offsets_1, offsets_2)
			pixel_loc = np.nonzero(im_depth>0)
			# im_labels = get_per_pixel_joints(im_depth, skel_pos)
			# pixel_labels = im_labels[pixel_loc]

			pred = rf.predict(features)
			im_predict = np.ones_like(im_depth)+N_SKEL_JOINTS+1
			im_predict[pixel_loc] = pred

			pos_pred = []
			from sklearn.cluster import MeanShift
			for i in xrange(N_SKEL_JOINTS):
				inds = np.nonzero(im_predict==i)
				if len(inds[0])>0:

					cluster_centers, cluster_counts = mean_shift(np.array(inds).T, 50, n_seeds=10, max_iterations=100)
					max_cluster = np.argmax(cluster_counts)
					pos_tmp = [cluster_centers[max_cluster].astype(np.int16)]
					pos_pred += [[pos_tmp[0][1], pos_tmp[0][0]]]
				else:
					pos_pred += [[-1]]

				# embed()
			# pos_pred = np.vstack(pos_pred)
			skel_pos_pred = np.zeros([N_MSR_JOINTS,2], dtype=np.int16)
			for i in xrange(len(SKEL_JOINTS)):
				if pos_pred[i][0] != -1: 
					skel_pos_pred[SKEL_JOINTS[i],:] = pos_pred[i]

			# Overlay skeletons
			if 1:
				im_predict = np.repeat(im_predict[:,:,None], 3, -1)
				im_predict = display_MSR_skeletons(im_predict, skel_pos, (20,), 'Upperbody')
				im_predict = display_MSR_skeletons(im_predict, skel_pos_pred, (0,20,0), 'Upperbody')
				max_ = (im_predict * (im_predict < 255)).max()

			all_pred_ims += [im_predict]

			''' Visualize '''
			if 1:
				# embed()
				# imshow((im_predict * (im_predict < 255)) / float(max_))				
				cv2.imshow("distances", (im_depth-im_depth.min()) / float(im_depth.max()-im_depth.min()))
				cv2.putText(im_predict, "Blue=Truth", (10, 210), cv2.FONT_HERSHEY_DUPLEX, .5, (int(im_predict.max()/2), 0, 0))
				cv2.putText(im_predict, "Green=Predict", (10, 230), cv2.FONT_HERSHEY_DUPLEX, .5, (0, int(im_predict.max()/2), 0))
				cv2.imshow("prediction", im_predict/im_predict.max().astype(np.float))
				# cv2.imshow("meanshift", im_mean/im_mean.max().astype(np.float))
				ret = cv2.waitKey(10)
				if ret > 0: break

		# except:
		# 	print "Frame failed:", i
		# 	break

	embed()	




if __name__ == '__main__':
	import optparse
	parser = optparse.OptionParser()
	parser.add_option('-l', '--learn', dest='learn', action="store_true", default=False, help='Enable skeleton')	
	parser.add_option('-i', '--infer', dest='infer', action="store_true", default=False, help='Enable skeleton')	
	(opt, args) = parser.parse_args()

	if opt.learn:
		main_learn()
	elif opt.infer:
		if len(args) > 0:
			filename = args[0]
		else:
			filename = None
		main_infer(filename)
	else:
		print "Enter -l or -i for learning or inference."





''' Parallelism '''
# from joblib import Parallel, delayed
# print "Computing with multiple threads"
# data = Parallel(n_jobs=-1)( delayed(compute_features)(n) for n in base_names )
# for n,i in zip(base_names, range(len(base_names))):
# 	if data[i] != -1:
# 		dataset_features[n] = data[i]