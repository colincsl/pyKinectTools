"""

Datatypes:

_skels_subactivity_ is a dictionary where the keys are subaction names.
Each subaction has a list of N time series (ndarray) iterations of that subaction
Format: {'a':[timeseries, timeseries,...], 'b':[timeseries, timeseries,...], ...}

_proto_motion_ is a dictionary where keys are subactions.
Each subaction has 1 timeseries (ndarray) for that subaction
Format: {'a':timeseries, 'b':timeseries, ...}


"""

import os
import time
import cPickle as pickle
import itertools as it
from copy import deepcopy

import numpy as np
import pandas as pd

import mlpy
import cv, cv2

from scipy.interpolate import *
from scipy.signal import resample
import scipy.misc as sm
from pylab import *

from pyKinectTools.utils.SkeletonUtils import *
from pyKinectTools.algs.DynamicTimeWarping import DynamicTimeWarping
from pyKinectTools.dataset_readers.CADPlayer import CADPlayer

# Debugging
from IPython import embed

CAD_DISABLED = [7,9,8,10,13,14]
CAD_ENABLED = [x for x in range(15) if x not in CAD_DISABLED]


def get_skels_per_subaction(skels, labels, min_seq_len=10):
	'''
	'''
	skel_segs = {}
	for i,lab in labels.items():
		start = lab['start']
		stop = lab['stop']
		if stop - start > min_seq_len:
			name = lab['subaction']
			if name in skel_segs.keys():
				skel_segs[name] += [skels[start:stop]]
			else:
				skel_segs[name] = [skels[start:stop]]
	
	return skel_segs

def get_rel_objects_per_affordance(skels, labels):
	'''
	'''
	skel_segs = {}
	for i,lab in labels.items():
		start = lab['start']
		stop = lab['stop']
		objects = lab['objects']
		for o in objects:
			name = objects[o]
			if name in skel_segs.keys():
				skel_segs[name] += [skels[start:stop,:,o]]
			else:
				skel_segs[name] = [skels[start:stop,:,o]]
	
	return skel_segs	
	
def split_skeleton_data(skels_subactivity, train_set=[1,3], test_set=[4]):
	'''
	Split skeletal data into train/test sets
	'''
	skels_train = {}
	for s in train_set:
		new_skels = deepcopy(skels_subactivity[s])
		for n in new_skels:
			if n not in skels_train:
				skels_train[n] = []
			skels_train[n] += new_skels[n]

	skels_test = {}
	for s in test_set:
		new_skels = deepcopy(skels_subactivity[s])
		for n in new_skels:
			if n not in skels_test:
				skels_test[n] = []
			skels_test[n] += new_skels[n]
	return skels_train, skels_test

def normalize_basis(skel):
	
	torso = skel[2]	
	shoulder1 = skel[3]
	shoulder2 = skel[5]
	midpoint = (shoulder1+shoulder2)/2
	# head = np.array([0,1,0])

	vec1 = shoulder1 - shoulder2
	vec1 /= np.linalg.norm(vec1)	
	vec2 = midpoint - torso
	vec2 /= np.linalg.norm(vec2)
	vec3 = np.cross(vec1, vec2)
	vec3 /= np.linalg.norm(vec3)
	basis = np.vstack([vec1, vec2, vec3])

	skel_new = skel.copy()
	for i_skel,s in enumerate(skel_new):
		skel_new[i_skel] = np.dot(basis, s-torso)# + torso

	return skel_new


def normalize_skel_stack(skels):
	'''
	Orients skeleton so the torso is centered and facing forwards
	Input: skels in skels_subactivity format
	'''
	for k in skels:
		for i,_ in enumerate(skels[k]):
			for ii,_ in enumerate(skels[k][i]):
				skel = skels[k][i][ii]
				skels[k][i][ii] = normalize_basis(skel)
	return skels



record = False
DIR = '/Users/colin/Data/CAD_120/'
framerate = 1
MAX_OBJECTS = 5
get_depth = True
''' -----------------TRAIN------------------- '''
# subactivities = []
skels_subactivity = {}
obj_subactivity = {}
object_affordances = {}
object_subactions = {}
cam = CADPlayer(base_dir=DIR, get_color=False, get_depth=get_depth, get_skeleton=True, 
				subjects=[1,3,4], actions=range(2), instances=[0,1,2])

# Only get the skeletal data
while cam.next(framerate):
	if cam.frame == 1:
		skels = cam.skel_stack['pos']
		tmp = get_skels_per_subaction(skels, cam.subactivity)
		if cam.subject not in skels_subactivity:
			skels_subactivity[cam.subject] = {}
		for t in tmp:
			if t not in skels_subactivity[cam.subject]:
				skels_subactivity[cam.subject][t] = []
			skels_subactivity[cam.subject][t] += tmp[t]
		cam.next_sequence()

# Get the skeletal data and the object positions
while cam.next(framerate):
	# Get joint angles
	if cam.frame == 1:
		skels = cam.skel_stack['pos']
		object_positions = []

	# Keep track of hands relative to objects
	if 1:#cam.frame%25==0:
		print "frame {} of {}".format(cam.frame, cam.framecount)

	obj_centroids = []
	obj_dists = []
	for o in cam.objects:
		tl = o['topleft']
		br = o['bottomright']
		ID = o['ID']
		if tl[0]==0 or br[0]==0:
			continue
		obj_im = cam.depthIm[tl[1]:br[1], tl[0]:br[0]]
		pt = np.array([[(tl[1]+br[1])/2, (tl[0]+br[0])/2, cam.depthIm[(tl[1]+br[1])/2, (tl[0]+br[0])/2]]])
		obj_centroids += [cam.camera_model.im2world(pt, [480,640])]
		obj_dists += [np.linalg.norm(pt-cam.users[0][2])]
		# print pt
	closest_object = np.argmin(obj_dists)
	object_positions += [obj_centroids[closest_object]]

	if cam.frame == cam.framecount:	
		object_positions = np.vstack(object_positions)
		#object_positions =  object_positions[:,:,None].repeat(skels.shape[0], 0).reshape([-1,3])
		# skels -= object_positions
		tmp = get_skels_per_subaction(skels, cam.subactivity)
		if cam.subject not in skels_subactivity:
			skels_subactivity[cam.subject] = {}
		for t in tmp:
			if t not in skels_subactivity[cam.subject]:
				skels_subactivity[cam.subject][t] = []
			skels_subactivity[cam.subject][t] += tmp[t]

		tmp = get_skels_per_subaction(object_positions, cam.subactivity)
		if cam.subject not in obj_subactivity:
			obj_subactivity[cam.subject] = {}
		for t in tmp:
			if t not in obj_subactivity[cam.subject]:
				obj_subactivity[cam.subject][t] = []
			obj_subactivity[cam.subject][t] += tmp[t]


	# # Keep track of hands relative to objects
	# if 1:#cam.frame%25==0:
	# 	print "frame {} of {}".format(cam.frame, cam.framecount)
	# for o in cam.objects:
	# 	tl = o['topleft']
	# 	br = o['bottomright']
	# 	ID = o['ID']
	# 	if tl[0]==0 or br[0]==0:
	# 		continue
	# 	obj_im = cam.depthIm[tl[1]:br[1], tl[0]:br[0]]
	# 	pt = np.array([[(tl[1]+br[1])/2, (tl[0]+br[0])/2, cam.depthIm[(tl[1]+br[1])/2, (tl[0]+br[0])/2]]])
	# 	obj_centroid = cam.camera_model.im2world(pt, [480,640])
	# 	# hand_distances += [[np.linalg.norm(obj_centroid - skel_orig[11]),
	# 	# 				np.linalg.norm(obj_centroid - skel_orig[12])]]
	# 	hand_distances[cam.frame-1, 0, ID] = np.linalg.norm(obj_centroid - skel_orig[11])
	# 	hand_distances[cam.frame-1, 1, ID] = np.linalg.norm(obj_centroid - skel_orig[12])
	# 	hand_distances[cam.frame-1, 2, ID] = np.linalg.norm(obj_centroid - skel_orig[2])
	# 	# hand_distances[cam.frame-1, 2, ID] = np.linalg.norm(obj_centroid - skel_orig[2])
	# 
	# if cam.frame == cam.framecount:
	# 	# embed()
	# 	tmp = get_rel_objects_per_affordance(hand_distances, cam.subactivity)
	# 	for t in tmp:
	# 		if t not in object_affordances:
	# 			object_affordances[t] = []
	# 		object_affordances[t] += tmp[t]
	# 	tmp = get_skels_per_subaction(hand_distances, cam.subactivity)
	# 	for t in tmp:
	# 		if t not in object_subactions:
	# 			object_subactions[t] = []
	# 		object_subactions[t] += tmp[t]


pickle.dump(skels_subactivity, open("/Users/colin/Desktop/skels_subactivity_1-2.dat", 'w'))
pickle.dump(obj_subactivity, open("/Users/colin/Desktop/objs_subactivity_1-2.dat", 'w'))

''' ---------------TEST--------------------- '''
skels_subactivity_test = {}
cam = CADPlayer(base_dir=DIR, get_color=False, get_depth=get_depth, get_skeleton=True, 
				subjects=[4], actions=[1,2], instances=[0,1,2])

prev_action = ''
while cam.next(framerate):
	# Get joint angles
	if cam.frame == 1:
		skels = cam.skel_stack['pos']		
		tmp = get_skels_per_subaction(skels, cam.subactivity)
		for t in tmp:
			if t not in skels_subactivity_test:
				skels_subactivity_test[t] = []
			skels_subactivity_test[t] += tmp[t]
		cam.next_sequence()

	if cam.frame == 1:
		skel_orig = cam.users[0].copy()

	# Change the subaction type
	if cam.subaction != prev_action:
		frame = 0
		print "a"
		prev_action = cam.subaction
	else:
		frame += 1

	# Get prototypical sequence
	i_iter = 0
	ang_frame = np.minimum(frame, len(skels_subactivity_angles[cam.subaction][i_iter])-1)
	angs = skels_subactivity_angles[cam.subaction][i_iter][ang_frame]

	# Get new motion from prototypical
	skel_orig = cam.users[0]
	skel_orig[:,1] *= -1
	centroid = cam.users[0][2]
	# centroid *= -1
	skel_new = get_CAD_skel_pos(skel_orig, angs, centroid)

	# Convert skel to image coordinates
	skel_orig_uv = cam.camera_model.world2im(skel_orig, [480,640])
	# skel_orig_uv[:,0] = 480 - skel_orig_uv[:,0]
	skel_new = cam.camera_model.world2im(skel_new, [480,640])
	# skel_new[:,0] = 480 - skel_new[:,0]

	# Look at objects
	objects_pos = []
	hand_distances = []
	# xyz_im = cam.camera_model.im2PosIm(cam.depthIm)
	# print xyz_im[skel_orig_uv[:,0], skel_orig_uv[:,1]], skel_orig

	for o in cam.objects:
		pt1 = tuple(o['topleft'].astype(np.int))
		pt2 = tuple(o['bottomright'].astype(np.int))
		if not all(pt1) or not all(pt2):
			continue
		cv2.rectangle(cam.depthIm, pt1, pt2, (0,0,0))
		pt1 = (pt1[0], pt1[1]+15)
		cv2.putText(cam.depthIm, cam.object_names[o["ID"]], pt1, cv2.FONT_HERSHEY_DUPLEX, .6, (0,0,0), thickness=1)


	for o in cam.objects:
		tl = o['topleft']
		br = o['bottomright']
		if tl[0]==0 or br[0]==0:
			continue
		# obj_im = xyz_im[tl[1]:br[1], tl[0]:br[0]]
		# obj_im = xyz_im[tl[0]:br[0], tl[1]:br[1]]
		obj_im = cam.depthIm[tl[1]:br[1], tl[0]:br[0]]
		# obj_centroid = xyz_im[(tl[1]+br[1])/2, (tl[0]+br[0])/2]
		pt = np.array([[(tl[1]+br[1])/2, (tl[0]+br[0])/2, cam.depthIm[(tl[1]+br[1])/2, (tl[0]+br[0])/2]]])
		# pt[0][1] = 640 - pt[0][1]
		# pt = np.array([[(tl[1]+br[1])/2, (tl[0]+br[0])/2, cam.depthIm[(tl[1]+br[1])/2, (tl[0]+br[0])/2]]])
		# pt[:,0] = 480 - pt[:,0]
		# obj_centroid = cam.camera_model.im2world(pt[:,[1,0,2]], [480,640])
		obj_centroid = cam.camera_model.im2world(pt, [480,640])
		# obj_centroid[:,1] *= -1
		objects_pos += [obj_centroid]
		# obj_centroid = np.mean(obj_im.reshape([-1,3]), 0)
		hand_distances += [[np.linalg.norm(obj_centroid - skel_orig[11]),
						np.linalg.norm(obj_centroid - skel_orig[12])]]
		scatter(cam.frame, hand_distances[-1][0])
		scatter(cam.frame, hand_distances[-1][1])
		cv2.line(cam.depthIm, ((tl[0]+br[0])/2, (tl[1]+br[1])/2), tuple(skel_orig_uv[12][[1,0]]), 0,thickness=2)
		cv2.line(cam.depthIm, ((tl[0]+br[0])/2, (tl[1]+br[1])/2), tuple(skel_orig_uv[11][[1,0]]), 0,thickness=2)
		print obj_centroid, skel_orig[11]
	print hand_distances


	# Vizualize
	# cam.depthIm = cam.depth_stack[0]*100
	# cam.depthIm = display_skeletons(cam.depthIm, skel_orig_uv, skel_type='CAD')
	cam.depthIm = display_skeletons(cam.depthIm, skel_new, skel_type='CAD', color=1000)
	print cam.frame, cam.subaction

	cam.visualize(color=False, depth=True)

skels_subactivity_train, skels_subactivity_test = split_skeleton_data([1,3], [4])

''' ------------------------------------- '''


''' Plot DTW for joint angles '''
for i_key,key in enumerate(skels_subactivity_angles):
	figure(key)
	print "{} iterations of {}".format(len(skels_subactivity_angles[key]), key)
	for i_iter in range(len(skels_subactivity_angles[key])):
		print 'iter', i_iter
		for i,ang in enumerate(skels_subactivity_angles[key][i_iter].T):
			x = skels_subactivity_angles[key][0][:,i]
			y = ang
			y = resample(y, len(x))
			# error, dtw_mat, y_ind = mlpy.dtw.dtw_std(x, y, dist_only=False)			
			error, dtw_mat, y_ind = DynamicTimeWarping(x, y)
			
			subplot(3,4,i+1)
			y_new = y[y_ind[0]]
			x_new = np.linspace(0, 1, len(y_new))
			# poly = polyfit(x_new, y_new, 5)
			# y_spline_ev = poly1d(poly)(x_new)

			nknots = 4
			idx_knots = (np.arange(1,len(x_new)-1,(len(x_new)-2)/np.double(nknots))).astype('int')
			knots = x_new[idx_knots]
			y_spline = splrep(x_new, y_new, t=knots)
			y_spline_ev = splev(np.linspace(0, 1, len(y_new)), y_spline)
			# plot(y_new)
			plot(y_spline_ev)
			y_spline_ev = resample(y_spline_ev, len(ang))
			skels_subactivity_angles[key][i_iter].T[i] = y_spline_ev
			# show()
			# plot(y[y_ind[0]])

			# subplot(3,10,i+1 + i_iter*10)
			# plot(x[y_ind[1]])
			# plot(y[y_ind[0]])
			print i,":", len(ang), len(x), len(y[y_ind[0]])
			title(CAD_JOINTS[i])

			if i == 10:
				break
show()


''' Plot relative object positions wrt to afforance '''
for i_key,key in enumerate(object_affordances):
	print "{} iterations of {}".format(len(object_affordances[key]), key)
	for i_iter in range(len(object_affordances[key])):
		print 'iter', i_iter
		for i,hand in enumerate(object_affordances[key][i_iter].T):
			y = hand
			y[y>2000] = 0
			subplot(2,8,i_key+1 + 8*i)
			plot(y)
			title(key)
show()

''' Plot relative object positions wrt to subaction '''
n_subactions = len(object_subactions.keys())
for i_key,key in enumerate(object_subactions):
	print "{} iterations of {}".format(len(object_subactions[key]), key)
	for i_iter in range(len(object_subactions[key])):
		print 'iter', i_iter
		for i,hand in enumerate(object_subactions[key][i_iter].T):
			y = hand[0]
			y[y>2000] = 0
			if np.all(y==0):
				continue
			subplot(2,n_subactions,i_key+1 + n_subactions*i- n_subactions)
			plot(y.T)
			title(key)
show()


''' Come up with prototypical motion for each subaction using DTW '''
def get_prototype_motions(skels_subactivity_train, smooth=False, nknots=10):
	'''
	Input: a set of skeleton trajectories
	Output: a motif/prototype skeleton trajectory

	This algorithm takes every instance of a class and compares it to every other instance
	in that class using DTW, optionally smooths. Each (pairwise) transformed class instance
	is then averaged to output a motif.

	Todo: currently this is done independently per-joint per-dimension. Should be per skeleton!
	'''
	skels_subactivity_train = normalize_skel_stack(skels_subactivity_train)

	proto_motion = {}
	for i_key,key in enumerate(skels_subactivity_train):
		n_instances = len(skels_subactivity_train[key])
		n_frames = int(np.mean([len(x) for x in skels_subactivity_train[key]]))
		# Do x,y,z seperately
		proto_motion[key] = np.zeros([n_frames, 15, 3])
		for i_joint in range(15):
			# error_matrix = np.zeros([n_instances, n_instances], np.float)
			y_spline_set = []
			for i in xrange(n_instances):
				for j in xrange(n_instances):
					if i >= j:
						continue
					x = skels_subactivity_train[key][i][:,i_joint]
					y = skels_subactivity_train[key][j][:,i_joint]
					# error, dtw_mat, y_ind = mlpy.dtw.dtw_std(x, y, dist_only=False)
					error, dtw_mat, y_ind = DynamicTimeWarping(x, y)
					# error = mlpy.dtw.dtw_std(x, y, dist_only=True)
					# error_matrix[i,j] = error

					y_new = y[y_ind[1]]
					x_new = np.linspace(0, 1, len(y_new))
					# poly = polyfit(x_new, y_new, 5)
					# y_spline_ev = poly1d(poly)(x_new)

					if smooth:
						# Generate Spline representation
						nknots = np.minimum(nknots, len(x_new)/2)
						idx_knots = (np.arange(1,len(x_new)-1,(len(x_new)-2)/np.double(nknots))).astype('int')
						knots = x_new[idx_knots]
						y_spline = splrep(x_new, y_new, t=knots)
						y_spline_ev = splev(np.linspace(0, 1, len(y_new)), y_spline)
						y_spline_ev = resample(y_spline_ev, n_frames)
						y_spline_set += [y_spline_ev]
					else:
						y_spline_ev = resample(y_new, n_frames)
						y_spline_set += [y_spline_ev]						

				proto_motion[key][:,i_joint] = np.mean(y_spline_set, 0)
	return proto_motion

''' Gaussian mixture model prototype '''
y_spline_set = np.vstack(y_spline_set)
from sklearn import gaussian_process
gp = gaussian_process.GaussianProcess(theta0=1e+1, normalize=False)
x = np.arange(y_spline_set.shape[1])[:,None].repeat(y_spline_set.shape[0], 1).T.astype(np.float)
x += np.random.random(x.shape)/10000
x_test = np.arange(y_spline_set.shape[1])[:,None]
y = y_spline_set - y_spline_set[:,0][:,None]
# gp.fit(y.ravel()[:,None], x.ravel()[:,None])
gp.fit(x.ravel()[:,None], y.ravel()[:,None])

y_pred, MSE = gp.predict(x_test, eval_MSE=True)
sigma = np.sqrt(MSE)
plot(x_test, y_pred, 'b', label=u'Prediction')
fill(np.concatenate([x_test, x_test[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plot(x_test, y[0], 'r', label=u'Obs')
for i in range(15):
	plot(x_test, y[i], 'r', label=u'Obs_DTW')
for i in range(6):
	plot(np.arange(len(skels_subactivity_train['opening'][i][:,-1,2])), skels_subactivity_train['opening'][i][:,-1,2], 'y', label=u'Obs')	
plot(proto_motion['opening'][:,-1,2], 'g', label=u'Prototype')
legend()
show()


''' Test similarity of samples to prototypical (eval on test data) '''

accuracy_dtw = []
accuracy_lcs = []
training_sets = list(it.combinations([1,3,4], 1))
testing_sets = [tuple([x for x in [1,3,4] if x not in y]) for y in training_sets]
for train_set, test_set in zip(training_sets, testing_sets):
	# Get training/test sets + calculate prototype motions
	skels_subactivity_train, skels_subactivity_test = split_skeleton_data(skels_subactivity, train_set, test_set)
	proto_motion = get_prototype_motions(skels_subactivity_train, nknots=5)
	skels_subactivity_test = normalize_skel_stack(skels_subactivity_test)
	print "Prototypes generated"

	# Generate precision/recall
	n_subactions = len(skels_subactivity_test.keys())
	max_iterations = max([len(skels_subactivity_test[x]) for x in skels_subactivity_test])
	errors_dtw = np.zeros([n_subactions, n_subactions,max_iterations], np.float)
	errors_lcs = np.zeros([n_subactions, n_subactions,max_iterations], np.float)
	errors_mask = np.zeros([n_subactions, max_iterations], dtype=np.int)
	# Evaluate each test instance
	for i_key,key in enumerate(skels_subactivity_test):
		n_instances = len(skels_subactivity_test[key])
		for i in xrange(n_instances):
			# Evaluate for each prototype
			for i_key2,key2 in enumerate(skels_subactivity_test):
				err_dtw = 0
				err_lcs = 0
				x_skel = proto_motion[key2]
				y_skel = skels_subactivity_test[key][i]

				for i_joint in CAD_ENABLED:
					for i_dim in range(3):
						x = x_skel[:,i_joint,i_dim]
						y = y_skel[:,i_joint,i_dim]
						error_dtw, _, y_ind = mlpy.dtw_std(x, y, dist_only=False, squared=True)
						error_lcs,_ = mlpy.lcs_real(x,y[y_ind[1]], np.std(x), len(x)/2)
						err_dtw += error_dtw
						err_lcs += error_lcs/len(x)
				errors_dtw[i_key, i_key2, i] = err_dtw
				errors_lcs[i_key, i_key2, i] = err_lcs
				errors_mask[i_key,i] = 1

	iterations_per_subaction = np.sum(errors_mask,1).astype(np.float)
	print "Train:{}, Test:{}".format(train_set, test_set) 
	solution = np.arange(n_subactions)[:,None].repeat(errors_dtw.shape[2], 1)
	true_positives = np.sum((errors_dtw.argmin(1) == solution)*errors_mask, 1)
	false_negatives = np.sum((errors_dtw.argmin(1) != solution)*errors_mask, 1)
	accuracy_dtw += [np.mean(true_positives / iterations_per_subaction)]
	print "DTW Precision:", accuracy_dtw[-1]
	# print "DTW Recall:", np.mean(true_positives / (true_positives+false_negatives).astype(np.float))
	# precision = True Positives / (True Positives + False Positives) = TP/TP+FP
	# recall = True Positives / (True Positives + False Negatives) = TP/TP+0



	accuracy_lcs += [np.mean(np.sum((errors_lcs.argmax(1) == solution)*errors_mask, 1) / iterations_per_subaction)]
	# print accuracy_lcs
	print "LCS Precision:", accuracy_lcs[-1]
	print ""

print '---- N-fold accuracy ---'
print "DTW: {:.4}%".format(np.mean(accuracy_dtw)*100)
print "LCS: {:.4}%".format(np.mean(accuracy_lcs)*100)


skels_subactivity = pickle.load(open("/Users/colin/Desktop/skels_subactivity_1-10.dat"))
# from sklearn.cross_validation import LeavePLabelOut
# data_splits = LeavePLabelOut([1,3,4], p=1, indices=True)
# train_set, test_set = list(data_splits)[0]

training_sets = list(it.combinations([1,3,4], 2))
testing_sets = [tuple([x for x in [1,3,4] if x not in y]) for y in training_sets]
train_set, test_set = zip(training_sets, testing_sets)[0]
skels_subactivity_train, skels_subactivity_test = split_skeleton_data(skels_subactivity, train_set, test_set)
obj_subactivity_train, obj_subactivity_test = split_skeleton_data(obj_subactivity, train_set, test_set)

skels_subactivity_test = normalize_skel_stack(skels_subactivity_test)
skels_subactivity_train = normalize_skel_stack(skels_subactivity_train)
proto_motion = get_prototype_motions(skels_subactivity_train, smooth=False)



''' Put together and visualize a new sequence '''
object_position = np.array([0, 1050,3500])
obj_position_uv = cam.camera_model.world2im(np.array([object_position]), [480,640])
# actions = ['null', 'moving', 'cleaning', 'moving', 'null', 'placing']
actions = proto_motion.keys()
a = actions[0]
new_action = proto_motion[a] + object_position
new_action_labels = [a]*len(new_action)
for a in actions:
	new_action = np.vstack([new_action, proto_motion[a] + object_position])
	new_action_labels += [a]*len(proto_motion[a])

from time import time
t0 = time()
ii = 0
sequence_samples = np.random.choice(n_samples, 5, replace=False)
for i,f in enumerate(new_action):
	if i>0 and new_action_labels[i] != new_action_labels[i-1]:
		ii = 0
		im = np.ones([480,640])*255
		n_samples = len(skels_subactivity_train[new_action_labels[i]])
		sequence_samples = np.random.choice(n_samples-1, 5, replace=False)
		cv2.imshow("New action", im)
		cv2.waitKey(1)

	bg_im = np.ones([480,640])
	# cv2.rectangle(bg_im, tuple(obj_position_uv[0][[1,0]]-[30,30]), tuple(obj_position_uv[0][[1,0]]+[30,30]), 2000)
	f_uv = cam.camera_model.world2im(f, [480,640])
	f_uv[:,0] = 480 - f_uv[:,0]
	im = display_skeletons(bg_im, f_uv, skel_type='CAD_Upper', color=2000)
	cv2.putText(im, "Action: "+new_action_labels[i], (20,60), cv2.FONT_HERSHEY_DUPLEX, 1, (2000,0,0), thickness=2)

	cv2.putText(im, "Prototype", (240,160), cv2.FONT_HERSHEY_DUPLEX, 1, (2000,0,0), thickness=2)
	# Plot training samples below the protype action
	for i_iter,i_sample in enumerate(sequence_samples):
		try:
			ii_frame = min(ii, len(skels_subactivity_train[new_action_labels[i]][i_sample])-1)
			skel = skels_subactivity_train[new_action_labels[i]][i_sample][ii_frame] - skels_subactivity_train[new_action_labels[i]][i_sample][ii_frame][2]
			# ii_frame = min(ii, len(skels_subactivity_test[new_action_labels[i]][i_iter])-1)
			# skel = skels_subactivity_test[new_action_labels[i]][i_iter][ii_frame] - skels_subactivity_test[new_action_labels[i]][i_iter][ii_frame][2]
			# skel = normalize_basis(skel)
			skel += [-1400+i_iter*700, 0,3500]
			f_uv = cam.camera_model.world2im(skel, [480,640])
			f_uv[:,0] = 480 - f_uv[:,0]
			im = display_skeletons(bg_im, f_uv, skel_type='CAD_Upper', color=2000)
		except: pass
	cv2.putText(im, "Training Samples: "+str(list(train_set)), (140,320), cv2.FONT_HERSHEY_DUPLEX, 1, (2000,0,0), thickness=2)

	# Plot test samples below the protype action
	for i_iter in range(5):
		try:
			ii_frame = min(ii, len(skels_subactivity_test[new_action_labels[i]][i_iter])-1)
			skel = skels_subactivity_test[new_action_labels[i]][i_iter][ii_frame] - skels_subactivity_test[new_action_labels[i]][i_iter][ii_frame][2]
			# skel = normalize_basis(skel)
			skel += [-1400+i_iter*700, -1000,3500]
			f_uv = cam.camera_model.world2im(skel, [480,650])
			f_uv[:,0] = 480 - f_uv[:,0]
			im = display_skeletons(bg_im, f_uv, skel_type='CAD_Upper', color=2000)
		except: pass
	cv2.putText(im, "Testing Samples: "+str(list(test_set)), (150,470), cv2.FONT_HERSHEY_DUPLEX, 1, (2000,0,0), thickness=2)

	cv2.imshow("New action", (im-1000.)/(im.max()-1000))
	cv2.waitKey(30)
	ii += 1
	print "{} fps".format(i/(time()-t0))



''' Spectral clustering of subaction instances '''
# 1) Generate similarity matrix
# for i_key,key in enumerate(skels_subactivity_test):
i_joint = 6
key = skels_subactivity_train.keys()[0]
n_instances = len(skels_subactivity_train[key])
similarity = np.empty([n_instances, n_instances], dtype=np.float)
for i in xrange(n_instances):
	x = skels_subactivity_train[key][i][:,i_joint]	
	for j in xrange(n_instances):
		# if i >= j:
			# continue
		y = skels_subactivity_train[key][j][:,i_joint]
		# error, dtw_mat, y_ind = mlpy.dtw.dtw_std(x, y, dist_only=False)
		error, dtw_mat, y_ind = DynamicTimeWarping(x, y)
		similarity[i,j] = error
	# plot(x[:,1], x[:,2], label=str(i))

similarity /= similarity.max()
W = 1. - similarity
D = np.eye(n_instances)*np.sum(W, 1)

if 0:
	# 2) Contrust graph laplacian (Unnormalized)
	L = D - W
else:
	# 2) Contrust graph laplacian (Normalized symmetric)
	Ds = D**(-.5) * np.eye(n_instances)
	Ds = np.nan_to_num(Ds)
	# L = I - D^-.5 S D^-.5
	L = np.eye(n_instances) - np.dot(np.dot(Ds,W),Ds)

# 3) Compute eigenvalues/vectors
eigs, eigvecs = np.linalg.eigh(L)

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from copy import deepcopy
COLOR='rgbyck'
best_bic = 0
best_clusters = []

# 4) Find the best K using BIC. Only use K vectors
for K in range(1, n_instances):
	sc = SpectralClustering(K)
	sc.fit(W)
	U = eigvecs[-K:].T # Shape n x K

	# 5) Cluster with kmeans or GMM. 
	clf = GMM(n_components=K)
	clf.fit(U)
	bic = clf.bic(U)
	if bic > best_bic:
		best_bic = bic
		best_clf = deepcopy(clf)
		best_clusters = best_clf.predict(U)		
	# print clf.predict(U)
	print "K={} BIC={}".format(K, clf.bic(U))

# Visualize
for i in xrange(n_instances):
	x = skels_subactivity_train[key][i][:,i_joint]
	figure(str(best_clusters[i]))
	plot(x[:,1], x[:,2], color=COLOR[best_clusters[i]])
show()

# Create one spline per component
n_knots = 4
for i in xrange(best_clusters.max()+1):

	idx_knots = (np.arange(1,len(x_new)-1,(len(x_new)-2)/np.double(n_knots))).astype('int')
	knots = x_new[idx_knots]
	y_spline = splrep(x_new, y_new, t=knots)
	y_spline_ev = splev(np.linspace(0, 1, len(y_new)), y_spline)
	# plot(y_new)
	plot(y_spline_ev)
	y_spline_ev = resample(y_spline_ev, len(ang))



'''
Baseline: Look at nearest neighbors for each subaction
'''

def fold_data(data, joints):
	'''
	Output all of the action samples as a single array
	'''
	all_data = np.concatenate([ np.array([data[k][i][:,joints] for i in xrange(len(data[k]))]) for k in data])
	labels = np.hstack([ np.array([k for i in xrange(len(data[k]))]) for k in data])
	return all_data, labels	

def fold_resample_data(data, joints, mean_length=None):
	'''
	Output all of the action samples as a single array and resample so they're the same length
	'''
	if mean_length is None:
		mean_length = np.mean(np.hstack([ [data[k][i].shape[0] for i in xrange(len(data[k]))] for k in data])).astype(np.int)
	all_data = np.concatenate([ np.concatenate([resample(data[k][i][:,joints], mean_length)[:,:,:,None] for i in xrange(len(data[k]))], 3) for k in data], 3)
	labels = np.hstack([ np.array([k for i in xrange(len(data[k]))]) for k in data])
	return all_data, labels
def fold_objects(data):
	'''
	Output all of the action samples as a single array and resample so they're the same length
	'''
	all_data = np.concatenate([ np.array([data[k][i] for i in xrange(len(data[k]))]) for k in data])
	# all_data = np.concatenate([ np.concatenate([data[k][i][:,None,:,None] for i in xrange(len(data[k]))], 3) for k in data], 3)
	labels = np.hstack([ np.array([k for i in xrange(len(data[k]))]) for k in data])
	return all_data, labels	
def fold_resample_objects(data, mean_length=None):
	'''
	Output all of the action samples as a single array and resample so they're the same length
	'''
	if mean_length is None:
		mean_length = np.mean(np.hstack([ [data[k][i].shape[0] for i in xrange(len(data[k]))] for k in data])).astype(np.int)
	all_data = np.concatenate([ np.concatenate([resample(data[k][i], mean_length)[:,None,:,None] for i in xrange(len(data[k]))], 3) for k in data], 3)
	labels = np.hstack([ np.array([k for i in xrange(len(data[k]))]) for k in data])
	return all_data, labels		

joints = [0,2,11,12]
n_joints = len(joints)
skel_train_data, all_train_labels = fold_data(skels_subactivity_train, joints)
obj_train_data, _ = fold_objects(obj_subactivity_train)
for i in range(skel_train_data.shape[0]):
	for j in range(n_joints):
		skel_train_data[i][:,j] -= obj_train_data[i]

skel_test_data, all_test_labels = fold_data(skels_subactivity_test, joints)
obj_test_data, _ = fold_objects(obj_subactivity_test)
for i in range(skel_test_data.shape[0]):
	for j in range(n_joints):
		skel_test_data[i][:,j] -= obj_test_data[i]

label_names = skels_subactivity_train.keys()
label_idx = {x:y for x,y in zip(label_names, xrange(len(label_names)))}
n_train_instances = skel_train_data.shape[-1]
n_test_instances = skel_test_data.shape[-1]
test_labeled = np.zeros(n_test_instances, np.int)

''' Calculate nearest neighbor using DTW distance'''
def nearest_neighbor_dtw(x, ys):
	similarity = np.zeros(n_train_instances, dtype=np.float)
	y_dims = ys[0].shape[1]*ys[0].shape[2]
	for y_data in skel_train_data:
		y = y_data.reshape([-1,y_dims])
		x = x.reshape([-1,y_dims])
		error,_ = DynamicTimeWarping(x, y, return_path=False)
		similarity[j] = error
	return np.argmin(similarity)

from joblib import Parallel, delayed
data = Parallel(n_jobs=-1)( delayed(nearest_neighbor_dtw)(x, ys) for x in )


ys = skel_train_data
for i in xrange(n_test_instances):
	# x = skel_test_data[i].reshape([-1,3*n_joints])
	test_labeled[i] = nearest_neighbor_dtw(x, ys)
	print "Truth:", all_test_labels[i], "Clf:", all_train_labels[np.argmin(similarity)]


for i in xrange(n_test_instances):
	for j in xrange(n_train_instances):
		x = skel_test_data[i].reshape([-1,3*n_joints])
		y = skel_train_data[j].reshape([-1,3*n_joints])
		error,_ = DynamicTimeWarping(x, y, return_path=False)
		similarity[j] = error
	test_labeled[i] = np.argmin(similarity)
	print "Truth:", all_test_labels[i], "Clf:", all_train_labels[np.argmin(similarity)]

test_labeled_name = [all_train_labels[x] for x in test_labeled]
dtw_accuracy = np.mean(test_labeled_name == all_test_labels)
print "Accuracy: {:0.3}%".format(dtw_accuracy*100.)

''' Calculate nearest neighbor using L2 distance'''
from sklearn.neighbors import NearestNeighbors
NN = NearestNeighbors(n_neighbors=1, metric='l2')

test_label_idx = [label_idx[x] for x in all_test_labels]
train_label_idx = [label_idx[x] for x in all_train_labels]

skel_train_data, all_train_labels = fold_resample_data(skels_subactivity_train, joints)
obj_train_data, _ = fold_resample_objects(obj_subactivity_train, mean_length=skel_train_data.shape[0])
skel_train_data -= obj_train_data

skel_test_data, all_test_labels = fold_resample_data(skels_subactivity_test, joints, mean_length=skel_train_data.shape[0])
obj_test_data, _ = fold_resample_objects(obj_subactivity_test, mean_length=skel_train_data.shape[0])
skel_test_data -= obj_test_data
# Train
X = skel_train_data.reshape([-1, skel_train_data.shape[-1]]).T
NN.fit(X)
# Test
X = skel_test_data.reshape([-1, skel_test_data.shape[-1]]).T
l = NN.kneighbors(X, 1, return_distance=False).T[0]
test_output = [train_label_idx[x] for x in l]
l2_accuracy = np.mean(np.equal(test_output, test_label_idx))
print "Accuracy: {:0.3}%".format(l2_accuracy*100)

# from sklearn.cross_validation import cross_val_score
# cross_val = cross_val_score(NN, X, test_label_idx)


''' Inverse kinematics from hand to torso? '''
''' Add physical/collision constraints '''
''' add symmetries '''
''' break into left hand, right hand, torso, legs '''

for i,f in enumerate(y_skel):
	bg_im = np.ones([480,640])
	# f_uv = cam.camera_model.world2im(f, [480,640])
	f_uv = cam.camera_model.world2im(f+[0,0,3000], [480,640])
	f_uv[:,0] = 480 - f_uv[:,0]
	im = display_skeletons(bg_im, f_uv, skel_type='CAD_Upper', color=2000)
	cv2.putText(im, "Action: "+new_action_labels[i], (20,60), cv2.FONT_HERSHEY_DUPLEX, 1, (2000,0,0), thickness=2)
	cv2.imshow("New action", (im-1000.)/(im.max()-1000))
	cv2.waitKey(1)

