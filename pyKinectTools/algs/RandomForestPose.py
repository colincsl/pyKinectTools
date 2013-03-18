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
TODO: add constaint to skeletons so they lie within the silhouette (learning!)
Projection to world space (3.2 pg 8 [PAMI])

Get rid of skeletons which don't change at all since last frame
add distance threshold to map pts to surface

'''

import os, time, pickle
import numpy as np
import scipy.misc as sm
import cv2
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from skimage import draw
from pyKinectTools.utils.SkeletonUtils import display_skeletons, kinect_to_msr_skel
from pyKinectTools.utils.DepthUtils import world2depth, depth2world, skel2depth
from pyKinectTools.dataset_readers.MSR_DailyActivities import read_MSR_depth_ims, read_MSR_color_ims, read_MSR_skeletons, read_MSR_labels, create_MSR_filenames
from pyKinectTools.algs.BackgroundSubtraction import extract_people
from pyKinectTools.algs.MeanShift import mean_shift
from pyKinectTools.algs.GeodesicSkeleton import generateKeypoints, distance_map
# from pyKinectTools.utils.VideoViewer import VideoViewer
from pyKinectTools.utils.KinectPlayer import KinectPlayer

from IPython import embed
from pylab import *
from skimage.draw import circle


N_MSR_JOINTS = 20
MSR_JOINTS = range(N_MSR_JOINTS)
WHOLE_JOINTS = [0, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19] # Low 
UPPER_JOINTS = [0, 3, 4, 5, 7, 8, 9, 11]

if 0:
	SKEL_JOINTS = UPPER_JOINTS
	SKEL_DISPLAY_MODE = "Upperbody"
if 0:
	SKEL_JOINTS = WHOLE_JOINTS
	SKEL_DISPLAY_MODE = "Kinect"	
if 1:
	SKEL_JOINTS = WHOLE_JOINTS
	SKEL_DISPLAY_MODE = "Low"
N_SKEL_JOINTS = len(SKEL_JOINTS)


''' ---------------- Learning ---------------- '''

class RFPose:

	def __init__(self, offset_max=125, feature_count=300, seed=0, parallelize=False):
		'''
		(Optional) offset_max
		(Optional) feature_count
		(Optional) seed
		'''

		self.offsets = create_rf_offsets(offset_max, feature_count, seed)
		self.features = []
		self.labels = []
		self.parallel = parallelize

	def add_frame(self, depth, skel):
		'''
		'''
		features, px_labels = learn_frame(depth, skel, self.offsets[0], self.offsets[1])
		self.features += [features]
		self.labels += [px_labels]

	def learn_forest(self):
		'''
		'''
		self.rf = RFClassifier(n_estimators=3,
							criterion='entropy',\
		 					max_depth=20, 
		 					max_features='auto',\
		  					oob_score=False,\
		  					n_jobs=-1, 
		  					random_state=None, 
		  					verbose=1,\
		  					min_samples_leaf=1)

		self.rf.fit(np.vstack(self.features), np.hstack(self.labels))
		save_data("Saved_Params/"+str(time.time())+"_forests.dat", {'rf':self.rf, 'offsets':self.offsets})
		print "Paramaters saved to: Saved_Params/{0:f}_forests.dat".format(time.time())


	def load_forest(self, filename=None):
		'''
		filename :  if None then loads the most recent in the Saved_Params folder
		'''
		if filename is None:
			files = os.listdir('Saved_Params/')
			filename = 'Saved_Params/' + files[-1]
			print "Loaded classifier file:",filename
	
		data =  pickle.load(open(filename))
		self.rf = data['rf']
		self.offsets = data['offsets']

	def infer_pose(self,depth):
		'''
		depth : depth image
		'''
		# Compute features and labels
		# embed()
		features = calculate_rf_features(depth, self.offsets[0], self.offsets[1])
		pixel_loc = np.nonzero(depth>0)

		pred = self.rf.predict(features)
		im_predict = np.ones_like(depth)+N_MSR_JOINTS+1
		im_predict[pixel_loc] = pred

		skel_pos_pred = pose_mean_shift(im_predict)


		''' Visualize '''
		if 1:
			cv2.imshow("prediction1", im_predict/float(im_predict.max()))
			im_predict = np.repeat(depth[:,:,None].astype(np.float), 3, -1)
			im_predict[depth>0] -= depth[depth>0].min()
			im_predict /= float(im_predict.max()/255.)
			im_predict = im_predict.astype(np.uint8)
			# im_predict = display_skeletons(im_predict, skel_pos, (255,0,0), SKEL_DISPLAY_MODE)
			im_predict = display_skeletons(im_predict, skel_pos_pred, (0,255,0), SKEL_DISPLAY_MODE)				

			cv2.putText(im_predict, "Blue=Truth", (10, 210), cv2.FONT_HERSHEY_DUPLEX, .5, (int(im_predict.max()/2), 0, 0))
			cv2.putText(im_predict, "Green=Predict", (10, 230), cv2.FONT_HERSHEY_DUPLEX, .5, (0, int(im_predict.max()/2), 0))
			cv2.imshow("prediction", im_predict)

			ret = cv2.waitKey(10)
			# if ret > 0: break



		return skel_pos_pred, im_predict		





''' ------------------------------------------ '''

def pose_mean_shift(im_predict):
	pos_pred = []
	label_set = np.unique(im_predict)
	for i in xrange(N_MSR_JOINTS):
		if i in label_set:
			inds = np.nonzero(im_predict==i)
			if len(inds[0])>0:
				X = np.array(inds).T
				cluster_centers, cluster_counts = mean_shift(X, 50, n_seeds=10, max_iterations=100)
				# Find cluster with most pixels
				max_cluster = np.argmax(cluster_counts)
				pos_tmp = [cluster_centers[max_cluster].astype(np.int16)]
				pos_pred += [[pos_tmp[0][1], pos_tmp[0][0]]]
			else:
				pos_pred += [[-1]]
		else:
			pos_pred += [[-1]]				

	skel_pos_pred = np.zeros([N_MSR_JOINTS,2], dtype=np.int16)
	for i in xrange(len(SKEL_JOINTS)):
		if pos_pred[i][0] != -1:
			skel_pos_pred[SKEL_JOINTS[i],:] = pos_pred[i]

	return skel_pos_pred


def create_rf_offsets(offset_max=125, feature_count=500, seed=0):
	'''
	Defaults are the max offset variability and feature count shown in [PAMI].
	feature_count should be 500
	offset_max should be 250

	From [PAMI] Sec 3.1, the second offsets list is 0 with probability 0.5
	'''
	np.random.seed(seed=seed)
	offsets_1 = (np.random.normal(0, 1, (feature_count,2)))*offset_max
	offsets_2 = (np.random.normal(0, 1, (feature_count,2)))*offset_max
	offsets_2[np.random.random(offsets_2.shape[0]) < 0.5] = 0

	return offsets_1, offsets_2


def calculate_rf_features(im, offsets_1, offsets_2, mask=None):
	'''
	im : masked depth image
	output : feature vector (n x len(offset))
	'''
	MAX_DIFF = 3000

	if mask is None:
		mask = im > 0

	# Get u,v positions for each pixel location
	pixels = np.nonzero((im*mask) > 0)
	px_count = pixels[0].shape[0]

	# Get depths of each pixel location
	depths = im[pixels]
	pixels = np.array(pixels).T
	n_features = len(offsets_1)

	output = np.zeros([px_count, n_features], dtype=np.int16)

	if px_count == 0:
		return output
	height, width = im.shape

	bg_mask = im==0
	im[bg_mask] = MAX_DIFF
	min_z = depths.min()
	im[-bg_mask] -= min_z
	# embed()
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
		dx = dx.astype(np.int16)

		in_bounds_y = (dy[:,0]>=0)*(dy[:,0]<height)*(dy[:,1]>=0)*(dy[:,1]<width)
		out_of_bounds_y = True - in_bounds_y
		dy[out_of_bounds_y] = [0,0]
		dy = dy.astype(np.int16)

		# Calculate actual offsets
		diffx = im[dx[:,0],dx[:,1]]
		diffy = im[dy[:,0],dy[:,1]]

		diff = diffx-diffy
		diff[out_of_bounds_y] = MAX_DIFF
		diff[out_of_bounds_x] = MAX_DIFF
		output[:,i] = diff

	im[-bg_mask] += min_z
	im[bg_mask] = 0
	return output


def pts_to_surface(skel, im_depth, thresh=10):
	'''
	Ensures that the joint positions lie within the silhouette of the person
	---Parameters---
	skel : should be in image coordinates
	im_depth : should be masked
	thresh : if distance is too large, don't move!
	---Return---
	The same skeleton where all joints are within the mask
	'''

	height, width = im_depth.shape
	skel = np.array([[max(min(p[0], width-1), 0), max(min(p[1], height-1), 0), p[2]] for p in skel] )
	out_of_bounds = np.where(np.array([im_depth[p[1],p[0]] for p in skel]) == 0)[0]

	# embed()
	# If pixel if outside of mask, find the closest 'in' neighbor
	if len(out_of_bounds) > 0:
		from sklearn.neighbors import NearestNeighbors  
		NN = NearestNeighbors(n_neighbors=1)
		inds = np.array(np.nonzero(im_depth)).T
		NN.fit(inds)

		for i in out_of_bounds:
			pos = skel[i]
			closest_ind = NN.kneighbors([pos[1],pos[0]], 1, return_distance=False)[0]
			closest_pos = inds[closest_ind][0]
			if np.linalg.norm(closest_pos[:2]-pos[:2], 2) < thresh:
				skel[i][0] = closest_pos[1]
				skel[i][1] = closest_pos[0]
			else:
				skel[i][0] = 0
				skel[i][1] = 0

	return skel

def get_per_pixel_joints(im_depth, skel, alg='geodesic', radius=5):
	'''
	Find the closest joint to each pixel using geodesic distances.
	---Paramaters---
	im_depth : should be masked depth image
	skel : in image coords
	alg : 'geodesic' or 'circular'
	radius : [only for circular]
	'''
	height, width = im_depth.shape
	distance_ims = np.empty([height, width, N_SKEL_JOINTS])
	MAX_VALUE = 32000

	# Get rid of edges around joints
	edge_thresh = 200#10
	gradients = np.gradient(im_depth)
	mag = np.sqrt(gradients[0]**2+gradients[1]**2)
	im_depth[mag>edge_thresh] = 0

	joints_out_of_bounds = np.abs(im_depth[skel[:,0], skel[:,1]] - skel[:,2]) > 100
	
	# Only look at major joints
	for i, j in zip(SKEL_JOINTS, range(N_SKEL_JOINTS)):
		if skel[i] in joints_out_of_bounds:
			distance_ims[:,:,j] = MAX_VALUE
			continue
		pos = skel[i]

		if alg == 'geodesic':
			x = np.maximum(np.minimum(pos[1], height-1), 0)
			y = np.maximum(np.minimum(pos[0], width-1), 0)
			# if j == 0:
			# 	pts, min_map = generateKeypoints(np.ascontiguousarray(im_depth), centroid=[x,y], iterations=10, scale=6)

			im_dist = distance_map(np.ascontiguousarray(im_depth), centroid=[x,y], scale=6)					
			distance_ims[:,:,j] = im_dist

			if 0:
				radius = 5
				x = np.maximum(np.minimum(pos[1], height-1-radius), radius)
				y = np.maximum(np.minimum(pos[0], width-1-radius), radius)
				# print x,y
				pts = circle(x,y, radius)
				# print pts[0]
				max_ = distance_ims[distance_ims[:,:,j]<MAX_VALUE, j].max()
				distance_ims[distance_ims[:,:,j]>=MAX_VALUE,j] = max_+100
				distance_ims[pts[0],pts[1],j] = max_+100
				figure(1)
				subplot(3,5,j+1)
				imshow(distance_ims[:,:,j])
				axis('off')
				# subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.1)
				tight_layout()
				subplots_adjust(left=.001, bottom=.001, wspace=.003, hspace=.003)

		elif alg == 'circular':
			x = np.maximum(np.minimum(pos[1], height-1-radius), radius)
			y = np.maximum(np.minimum(pos[0], width-1-radius), radius)
			pts = draw.circle(x,y, radius)
			closest_pos[pts] = i
		else:
			print "No algorithm type in 'get_per_pixel_joints'"


	if alg == 'geodesic':
		closest_pos = np.argmin(distance_ims, -1).astype(np.uint8)
		closest_pos_all = np.argsort(distance_ims, -1).astype(np.uint8)
	elif alg == 'circular':
		closest_pos = np.zeros_like(im_depth)
	
	#Change all background pixels
	# mask = im_depth!=0
	# mask = (distance_ims.max(-1)<32000)
	mask = im_depth!=0
	closest_pos[-mask] = N_MSR_JOINTS+1

	if 0:
		# Reorder for occlusions
		skel_depths = np.array([skel[i,2] for i in SKEL_JOINTS])
		skel_XYZ = np.array([skel[i] for i in SKEL_JOINTS])
		pos_order = np.argsort(skel_depths).tolist()
		pos_order.reverse()

		residual = im_depth[mask] - skel_depths[closest_pos[mask]]
		im_tmp = np.zeros_like(im_depth)
		im_tmp[mask] = residual
		closest_arg = np.argmin(distance_ims+skel_depths, -1)
		closest_arg[-mask] = -1


		# If geo dist < euc distance then kill
		# Go from back to front. Only care about things ahead
		from scipy.spatial import distance
		# euclid_dists = distance.squareform(distance.pdist(depth2world(skel_XYZ, [240,320])))
		euclid_dists = distance.squareform(distance.pdist(skel_XYZ))
		geo_dists = np.zeros_like(euclid_dists)
		
		for i in xrange(len(SKEL_JOINTS)):
			p = pos_order[i]
			# If a joint is behind it, don't care -- set to infinity
			geo_dists[p,:] = [distance_ims[skel[j][0],skel[j][1],p] if j in pos_order[i:] else np.inf for j in SKEL_JOINTS]

		# Check if it's incorrect at least 2x
		removed_joints = []
		err = (geo_dists*2. < euclid_dists)
		for i in xrange(len(err)):
			if err[i].sum(0) >= 1:
				# removed_joints += [i]
				distance_ims[:,:,i] = MAX_VALUE

		closest_pos_e = np.argmin(distance_ims, -1).astype(np.uint8)
		closest_pos_e[-mask] = N_MSR_JOINTS+1

		subplot(2,2,1)
		closest_pos = display_skeletons(closest_pos, skel, color=(23,), skel_type='Low')
		imshow(closest_pos); axis('off')
		subplot(2,2,2)
		imshow(closest_pos_e); axis('off')
		subplot(2,2,3)
		imshow(closest_pos_e==closest_pos); axis('off')
		# blank = np.ones(300,300)
		# removed_text = ''

		# for i in removed_joints:
			# removed_text += str(i)+" "
			# cv2.putText()
		tight_layout()
		subplots_adjust(left=.001, bottom=.001, wspace=.003, hspace=.003)
		show()

	# embed()
	
	# closest_pos_e[im_depth==0] = N_MSR_JOINTS+1

	# closest_pos2[-mask] = N_MSR_JOINTS+1
	# closest_pos2[closest_pos>=32000] = N_MSR_JOINTS+1
	# closest_arg[im_depth==0] = N_MSR_JOINTS+1

	# figure(1); imshow(closest_arg);
	# figure(3); imshow(closest_pos2)
	
	# # Reorder
	# im_order = np.zeros_like(im_depth)-1
	# for p in range(len(pos_order)):
	# 	mask_tmp = closest_pos==p
	# 	dists_tmp = distance_ims[mask_tmp, p] + skel[pos_order[p],2]
	# 	im_order[mask_tmp] = np.maximum(im_order[mask_tmp], dists_tmp)

	# figure(100)
	# imshow(closest_pos)
	# figure(101)
	# imshow(im_depth)	
	# show()
	# embed()
	return closest_pos


def learn_frame(im_depth, skel_pos, offsets_1, offsets_2):

	''' Get frame data '''
	skel_pos = pts_to_surface(skel_pos, im_depth, thresh=50)
	
	''' Compute features and labels '''
	im_labels = get_per_pixel_joints(im_depth, skel_pos, 'geodesic')
	im_labels[(im_depth==0)*(im_labels>N_MSR_JOINTS)] = N_MSR_JOINTS+1
	# im_labels[(im_depth>0)*(im_labels==255)] = N_MSR_JOINTS+1
	mask = (im_labels<N_MSR_JOINTS)
	pixel_loc = np.nonzero(mask)
	pixel_labels = im_labels[pixel_loc]	
	# embed()
	features = calculate_rf_features(im_depth, offsets_1, offsets_2, mask=mask)

	''' Visualize '''
	if 1:
		im_labels = np.repeat(im_labels[:,:,None], 3, -1)
		im_labels = display_skeletons(im_labels, skel_pos, (20,), skel_type='Low')
		cv2.putText(im_labels, "Blue=Truth", (10, 210), cv2.FONT_HERSHEY_DUPLEX, .5, (int(im_labels.max()/2), 0, 0))
		cv2.putText(im_labels, "Green=Predict", (10, 230), cv2.FONT_HERSHEY_DUPLEX, .5, (0, int(im_labels.max()/2), 0))					
		cv2.imshow("feature_space", im_labels/im_labels.max().astype(np.float))
		ret = cv2.waitKey(10)
		if ret > 0: exit

	return features, pixel_labels


def learn_msr_video(name, offsets_1, offsets_2):
	'''
	'''
	depth_file = name + "depth.bin"
	color_file = name + "rgb.avi"
	skeleton_file = name + "skeleton.txt"
	''' Read data from each video/sequence '''
	try:
		depthIms, maskIms = read_MSR_depth_ims(depth_file)
		depthIms *= maskIms
		depthIms /= 10
		colorIms = read_MSR_color_ims(color_file)
		skels_world, _ = read_MSR_skeletons(skeleton_file)
		skels_world[:,2]/=10
	except:
		print "Error getting frame features"
		return -1,-1

	all_features = []
	all_labels = []
	for i in xrange(0, len(depthIms), 25):
		im_depth = depthIms[i]
		skel_pos = skel2depth(skels_world[i], rez=[240,320])
		# ''' --- NOTE THIS 10 PX OFFSET IN THE MSR DATASET !!! --- '''
		# skel_pos[:,0] -= 10

		features, pixel_labels = learn_frame(im_depth, skel_pos, offsets_1, offsets_2)
		''' Stack features '''
		all_features += [features]
		all_labels += [pixel_labels]

	return np.concatenate(all_features), np.concatenate(all_labels)
	# return all_features, all_labels

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

	''' My data '''
	if 1:
		# VP = KinectPlayer(base_dir='/Users/colin/Data/Office_25Feb2013/', device=2, get_depth=True, get_color=False, get_skeleton=True, get_mask=False)
		# VP = KinectPlayer(base_dir='/Users/colin/Data/Room_close_26Feb13/', device=1, get_depth=True, get_color=False, get_skeleton=True, get_mask=False)
		
		# Scrap first 50 frames
		# depthIms, skels_world = VP.get_n_skeletons(50)
		depthIms, skels_world = VP.get_n_skeletons(400)
		# depthIms, skels_world = VP.get_n_skeletons(-1)
		print "{0:d} frames loaded".format(len(depthIms))
		# for i in range(0, len(depthIms), 1):
		for i in range(0, 40):
			# try:
			if 1:
				print "Frame", i
				depthIms, skels_world = VP.get_n_skeletons(10)
				if depthIms == -1:
					break
				depth = depthIms[0]
				skel_w = skels_world[0]
				# depth = depthIms[i]
				# skel_w = skels_world[i]
				skel_im = skel2depth(skel_w, rez=[240,320])
				features, pixel_labels = learn_frame(depth, skel_im, offsets_1, offsets_2)

				all_features += [features]
				all_labels += [pixel_labels]
			# except:
				# print "Error in frame {0:d}".format(i)
		del depthIms, skels_world

	''' MSR data'''
	if 0:
		names = create_MSR_filenames(np.arange(1)+1, np.arange(3, 5)+1, [2])
		# Parallelize feature collection
		from joblib import Parallel, delayed
		if 1:
			for n_set in chunks(names, 1):
				print "Computing with one thread. Current feature count:", len(all_features)
				all_data = learn_msr_video(n_set[0], offsets_1, offsets_2)
				if all_features == []:
					all_features = all_data[0]
					all_labels = all_data[1]
				else:
					all_features = np.vstack([all_features, all_data[0]])
					all_labels = np.hstack([all_labels, all_data[1]])
			
		else:
			for n_set in chunks(names, 100):		
				print "Computing with multiple threads. Current feature count:", len(all_features)
				all_data = Parallel(n_jobs=-1, verbose=True, pre_dispatch=2)( delayed(learn_frame)(n, offsets_1, offsets_2) for n in n_set )
				# Account for bad frames
				all_features += [f[0] for f in all_data if f[0] is not -1]
				all_labels += [f[1] for f in all_data if f[1] is not -1]
				print "Done computing this set of features/labels"

	print "Done computing all features/labels"
	all_features = np.vstack(all_features)
	all_labels = np.hstack(all_labels)

	print "Starting forest"
	rf = RFClassifier(n_estimators=6,
						criterion='entropy',\
	 					max_depth=20, 
	 					max_features='auto',\
	  					oob_score=False,\
	  					n_jobs=-1, 
	  					random_state=None, 
	  					verbose=1,\
	  					min_samples_leaf=1)

	rf.fit(all_features, all_labels)

	save_data("Saved_Params/"+str(time.time())+"_forests.dat", {'rf':rf, 'offsets':[offsets_1, offsets_2]})


def save_data(name, data):
	with open(name, 'w') as f:
	 	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


''' ---------------- Inference ---------------- '''

def infer_pose(im_depth, rf, offsets_1, offsets_2):
	'''
	rf : pretrained random forest
	'''
	''' Compute features and labels '''
	features = calculate_rf_features(im_depth, offsets_1, offsets_2)
	pixel_loc = np.nonzero(im_depth>0)

	pred = rf.predict(features)
	im_predict = np.ones_like(im_depth)+N_MSR_JOINTS+1
	im_predict[pixel_loc] = pred

	''' Mean shift '''
	pos_pred = []
	label_set = np.unique(im_predict)
	for i in xrange(N_MSR_JOINTS):
		if i in label_set:
			inds = np.nonzero(im_predict==i)
			if len(inds[0])>0:
				X = np.array(inds).T
				cluster_centers, cluster_counts = mean_shift(X, 50, n_seeds=10, max_iterations=100)
				# Find cluster with most pixels
				max_cluster = np.argmax(cluster_counts)
				pos_tmp = [cluster_centers[max_cluster].astype(np.int16)]
				pos_pred += [[pos_tmp[0][1], pos_tmp[0][0]]]
			else:
				pos_pred += [[-1]]
		else:
			pos_pred += [[-1]]				

	skel_pos_pred = np.zeros([N_MSR_JOINTS,2], dtype=np.int16)
	for i in xrange(len(SKEL_JOINTS)):
		if pos_pred[i][0] != -1:
			skel_pos_pred[SKEL_JOINTS[i],:] = pos_pred[i]

	return skel_pos_pred, im_predict

def main_infer(rf_name=None):
	'''
	'''

	if rf_name is None:
		import os
		files = os.listdir('Saved_Params/')
		rf_name = 'Saved_Params/' + files[-1]
		print "Classifier file:",rf_name


	# Load classifier data
	data =  pickle.load(open(rf_name))
	rf = data['rf']
	offsets_1, offsets_2 = data['offsets']

	''' Read data from each video/sequence '''
	depthIms = []
	skels_world = []
	if 1:
		# VP = KinectPlayer(base_dir='/Users/colin/Data/Office_25Feb2013/', device=2, get_depth=True, get_color=False, get_skeleton=True, get_mask=False)	
		VP = KinectPlayer(base_dir='/Users/colin/Data/Room_close_26Feb13/', device=1, get_depth=True, get_color=False, get_skeleton=True, get_mask=False)	
		_,_ = VP.get_n_skeletons(50)
		depthIms, skels_world = VP.get_n_skeletons(100)
		# depthIms = np.array(depthIms)[:,:,::-1]

	else:
		# name = 'a01_s01_e02_'
		name = 'a01_s10_e02_'
		# name = 'a02_s06_e02_'
		# name = 'a05_s02_e02_'
		depth_file = name + "depth.bin"
		color_file = name + "rgb.avi"
		skeleton_file = name + "skeleton.txt"

		try:
			depthIms, maskIms = read_MSR_depth_ims(depth_file)
			depthIms *= maskIms
			depthIms /= 10
			# colorIms = read_MSR_color_ims(color_file)
			skels_world, skels_im = read_MSR_skeletons(skeleton_file)
			skels_world[:,2]/=10
		except:
			print "Error reading data"


	''' Process data '''
	all_pred_ims = []
	for i in xrange(1, len(depthIms), 1):
		# try:
		if 1:
			print i
			''' Get frame data '''
			im_depth = depthIms[i]
			# im_depth[160:, :] = 0
			skel_pos = skel2depth(skels_world[i], rez=[240,320])
			# ''' --- NOTE THIS 10 PX OFFSET IN THE MSR DATASET !!! --- '''
			# skel_pos[:,0] -= 10

			skel_pos_pred, im_predict = infer_pose(im_depth, rf, offsets_1, offsets_2)

			# Overlay skeletons
			if 1:
				# colorIm = colorIms[i]
				# im_predict = colorIm
				cv2.imshow("forest_prediction", im_predict/float(im_predict.max()))

				im_predict = np.repeat(im_depth[:,:,None].astype(np.float), 3, -1)
				# embed()
				im_predict[im_depth>0] -= im_depth[im_depth>0].min()
				im_predict /= float(im_predict.max()/255.)
				im_predict = im_predict.astype(np.uint8)


				# im_predict = np.repeat(im_predict[:,:,None], 3, -1)
				# im_predict /= float(im_predict.max())*255
				# im_predict = im_predict.astype(np.uint8)
				im_predict = display_skeletons(im_predict, skel_pos, (255,0,0), SKEL_DISPLAY_MODE)
				im_predict = display_skeletons(im_predict, skel_pos_pred, (0,255,0), SKEL_DISPLAY_MODE)
				im_predict = display_skeletons(im_predict, skel_pos, (255,0,0), SKEL_DISPLAY_MODE)
				im_predict = display_skeletons(im_predict, skel_pos_pred, (0,255,0), SKEL_DISPLAY_MODE)				

				# embed()
				# max_ = (im_predict * (im_predict < 255)).max()

			all_pred_ims += [im_predict]

			''' Visualize '''
			if 1:
				cv2.putText(im_predict, "Blue=Truth", (10, 210), cv2.FONT_HERSHEY_DUPLEX, .5, (int(im_predict.max()/2), 0, 0))
				cv2.putText(im_predict, "Green=Predict", (10, 230), cv2.FONT_HERSHEY_DUPLEX, .5, (0, int(im_predict.max()/2), 0))
				cv2.imshow("prediction", im_predict)

				ret = cv2.waitKey(10)
				if ret > 0: break

				time.sleep(.5)

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
