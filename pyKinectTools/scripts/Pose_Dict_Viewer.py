"""
Main file for training multi-camera pose
"""

#import os
#import time
import itertools as it
from joblib import Parallel, delayed
import cPickle as pickle
import optparse
from copy import deepcopy

import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd
import Image
import cv2

import skimage
from skimage import color
from skimage.draw import line, circle
from skimage.color import rgb2gray,gray2rgb, rgb2lab
from skimage.feature import local_binary_pattern, match_template, peak_local_max

from pyKinectTools.utils.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.utils.DepthUtils import *
from pyKinectTools.utils.SkeletonUtils import display_skeletons, transform_skels, kinect_to_msr_skel, msr_to_kinect_skel
from pyKinectTools.dataset_readers.MHADPlayer import MHADPlayer
from pyKinectTools.algs.GeodesicSkeleton import *
from pyKinectTools.algs.PoseTracking import *
from pyKinectTools.algs.LocalOccupancyPattern import *
from pyKinectTools.algs.IterativeClosestPoint import IterativeClosestPoint

from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import AdditiveChi2Sampler

from IPython import embed
np.seterr(all='ignore')

# -------------------------MAIN------------------------------------------

def main(visualize=False, learn=False, actions=None, subjects=None, n_frames=220):
	# learn = True
	# learn = False
	if actions is []:
		actions = [2]
	if subjects is []:
		subjects = [2]
	# actions = [1]
	# actions = [1, 2, 3, 4, 5]
	# subjects = [1]
	if 1:
		MHAD = True
		cam = MHADPlayer(base_dir='/Users/colin/Data/BerkeleyMHAD/', kinect=1, actions=actions, subjects=subjects, reps=[1], get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
	else:
		MHAD = False
		cam = KinectPlayer(base_dir='./', device=2, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
		bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/CIRL_Background_A.tif')
		bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/CIRL_Background_B.tif')
		cam.bgSubtraction.backgroundModel = np.array(bg.getdata()).reshape([240,320]).clip(0, 4500)
	height, width = cam.depthIm.shape
	skel_previous = None

	# clf_geo = pickle.load(open('geodesic_svm_sorted_scaled_5class.pkl'))
	# clf_color,color_approx = pickle.load(open('color_histogram_approx_svm_5class.pkl'))
	# clf_lbp,lbp_approx = pickle.load(open('lbp_histogram_approx_svm_5class.pkl'))

	face_detector = FaceDetector()
	hand_detector = HandDetector(cam.depthIm.shape)
	curve_detector = CurveDetector(cam.depthIm.shape)

	# Video writer
	# video_writer = cv2.VideoWriter("/Users/colin/Desktop/test.avi", cv2.cv.CV_FOURCC('M','J','P','G'), 15, (320,240))

	# Save Background model
	# im = Image.fromarray(cam.depthIm.astype(np.int32), 'I')
	# im.save("/Users/Colin/Desktop/k2.png")

	# Setup pose database
	append = True
	append = False
	pose_database = PoseDatabase("PoseDatabase.pkl", learn=learn, search_joints=[0,4,7,10,13], append=append)

	# Per-joint classification
	head_features = []
	hand_features = []
	feet_features = []
	joint_features = {'geodesic':[None]*14,
					'color_histograms':[None]*14,
					'lbp':[None]*14}

	# Evaluation
	accuracy_all = []
	joint_accuracy_all = []
	geo_accuracy = []
	color_accuracy = []
	lbp_accuracy = []

	frame_count = 0
	frame_rate = 2
	if not MHAD:
		cam.next(350)
	frame_prev = 0
	try:
	# if 1:
		while cam.next(frame_rate):# and frame_count < n_frames:
			if frame_count - frame_prev > 100:
				print ""
				print "Frame #{0:d}".format(frame_count)
				frame_prev = frame_count

			if not MHAD:
				if len(cam.users) == 0:
					continue
				else:
					# cam.users = [np.array(cam.users[0]['jointPositions'].values())]
					if np.any(cam.users[0][0] == -1):
						continue
					cam.users[0][:,1] *= -1
					cam.users_uv_msr = [cam.camera_model.world2im(cam.users[0], [240,320])]

			# Apply mask to image
			if MHAD:
				mask = cam.get_person(2) > 0
			else:
				mask = cam.get_person() > 0
				if np.all(mask==False):
					continue

			im_depth =  cam.depthIm
			cam.depthIm[cam.depthIm>3000] = 0
			im_color = cam.colorIm*mask[:,:,None]
			cam.colorIm *= mask[:,:,None]
			pose_truth = cam.users[0]
			pose_truth_uv = cam.users_uv_msr[0]

			# Get bounding box around person
			box = nd.find_objects(mask)[0]
			d = 20
			# Widen box
			box = (slice(np.maximum(box[0].start-d, 0), \
					np.minimum(box[0].stop+d, height-1)), \
				   slice(np.maximum(box[1].start-d, 0), \
					np.minimum(box[1].stop+d, width-1)))
			box_corner = [box[0].start,box[1].start]

			''' ---------- ----------------------------------- --------'''
			''' ----------- Feature Detector centric approach ---------'''
			''' ---------- ----------------------------------- --------'''

			''' ---- Calculate Detectors ---- '''
			# Face detection
			face_detector.run(im_color[box])
			# Skin detection
			hand_markers = hand_detector.run(im_color[box], n_peaks=3)
			# curve detection
			# curve_markers = curve_detector.run((im_depth*mask)[box], n_peaks=3)
			# Calculate LBPs ##Max P=31 for LBPs becuase of datatype
			# x = local_occupancy_pattern(cam.depthIm[box]*mask[box], [5,5,5],[3,3,3])
			# lop_texture = local_binary_pattern_depth(cam.depthIm[box]*mask[box], 10, 20, px_diff_thresh=100)*mask[box]
			# lop_markers = []#peak_local_max(lop_texture, min_distance=20, num_peaks=5, exclude_border=False)
			# lbp_texture = local_binary_pattern(cam.depthIm[box]*mask[box], 6, 20)*mask[box]
			# Calculate Geodesic Extrema
			im_pos = cam.camera_model.im2PosIm(cam.depthIm*mask)[box] * mask[box][:,:,None]
			geodesic_markers = geodesic_extrema_MPI(im_pos, iterations=5, visualize=False)
			# geodesic_markers, geo_map = geodesic_extrema_MPI(im_pos, iterations=5, visualize=True)
			geodesic_markers_pos = im_pos[geodesic_markers[:,0], geodesic_markers[:,1]]

			markers = list(geodesic_markers) + list(hand_markers) #+ list(lop_markers) + curve_markers
			markers = np.array([list(x) for x in markers])

			if 1:
				''' ---- Database lookup ---- '''
				pts_mean = im_pos[(im_pos!=0)[:,:,2]].mean(0)
				if learn:
					# Normalize pose
					pose_uv = cam.users_uv[0]
					if np.any(pose_uv==0):
						print "skip"
						frame_count += frame_rate
						continue
					# print pose_truth[2], pts_mean
					pose_database.update(pose_truth - pts_mean)

				else:
					# Concatenate markers
					markers = list(geodesic_markers) + hand_markers
					# markers = list(geodesic_markers) + list(lop_markers) + curve_markers + hand_markers
					markers = np.array([list(x) for x in markers])

					# Normalize pose
					pts = im_pos[markers[:,0], markers[:,1]]
					pts = np.array([x for x in pts if x[0] != 0])
					pts -= pts_mean

					# Get closest pose
					pose = pose_database.query(pts, knn=1)
					# pose = pose_database.weighted_query(pts, knn=1)

					# pose = pose_database.reverse_query(pts[:,[1,0,2]])

					# im_pos -= pts_mean
					# R,t = IterativeClosestPoint(pose, im_pos.reshape([-1,3])-pts_mean, max_iters=5, min_change=.001, pt_tolerance=10000)
					# pose = np.dot(R.T, pose.T).T - t
					# pose = np.dot(R, pose.T).T + t

					pose += pts_mean
					pose_uv = cam.camera_model.world2im(pose, cam.depthIm.shape)

					# Constrain
					if 0:
						try:
							''' This does worse because the joint may fall to a different part of the body (e.g. hand to torso) which throws the error upward '''

							surface_map = nd.distance_transform_edt(im_pos[:,:,2]==0, return_distances=False, return_indices=True)
							pose_uv[:,:2] = surface_map[:, pose_uv[:,0]-box_corner[0], pose_uv[:,1]-box_corner[1]].T + [box_corner[0], box_corner[1]]
							pose = cam.camera_model.im2world(pose_uv)

							# skel_current = link_length_constraints(skel_current, constraint_links, constraint_values, alpha=.5)
							# skel_current = geometry_constraints(skel_current, joint_size, alpha=0.5)
							# skel_current = collision_constraints(skel_current, constraint_links)
							# embed()
							# pose_uv_box = pose_uv - [box_corner[0], box_corner[1], 0]
							# pose_uv_box = pose_uv_box.clip([0,0,0], [cam.depthIm.shape[0]-1, cam.depthIm.shape[1]-1, 9999])
							# joint_size = np.array([75]*14)
							# pose_n, pose_uv_n = ray_cast_constraints(pose, pose_uv_box, im_pos, surface_map, joint_size)
							# print 'Pose',pose,pose_n
							# pose = pose_n
							# pose_uv = pose_uv_n + [box_corner[0], box_corner[1], 0]

						except:
							print 'error constraining'

					# skel_previous = np.array(pose, copy=True)

			display_markers(cam.colorIm, hand_markers[:2], box, color=(0,250,0))
			if len(hand_markers) > 2:
				display_markers(cam.colorIm, [hand_markers[2]], box, color=(0,200,0))
			display_markers(cam.colorIm, geodesic_markers, box, color=(200,0,0))
			# display_markers(cam.colorIm, curve_markers, box, color=(0,100,100))
			# display_markers(cam.colorIm, lop_markers, box, color=(0,0,200))

			if 0:
				''' ---------- ----------------------------------- --------'''
				''' ---------- Feature Descriptor centric approach --------'''
				''' ---------- ----------------------------------- --------'''
				''' ---- Calculate Descriptors ---- '''
				hand_markers = np.array(hand_markers)
				# Geodesics
				geodesic_features = relative_marker_positions(im_pos, geodesic_markers_pos[:,[1,0,2]])
				geodesic_features = np.sort(geodesic_features)
				# Color Histogram
				skin = skimage.exposure.rescale_intensity(hand_detector.im_skin, out_range=[0,255]).astype(np.uint8)
				color_histograms = local_histograms(skin, n_bins=5, max_bound=255, patch_size=11)*mask[box][:,:,None]
				# LBP Histogram
				lbp_texture = local_binary_pattern(cam.depthIm[box]*mask[box], 6, 5)*mask[box]
				lbp_histograms = local_histograms(lbp_texture.astype(np.uint8), n_bins=10, max_bound=2**6, patch_size=11)*mask[box][:,:,None]
				# for i in range(10):
				# 	subplot(2,5,i+1)
				# 	imshow(lbp_histograms[:,:,i])

				''' ---- Per Joint Learning ---- '''
				if learn:
					for ii,i in enumerate(pose_truth_uv):
						if i[0] != 0:
							try:
								if joint_features['geodesic'][ii] is None:
									joint_features['geodesic'][ii] = geodesic_features[i[1]-box_corner[0], i[0]-box_corner[1]]
								else:
									joint_features['geodesic'][ii] = np.vstack([joint_features['geodesic'][ii], (geodesic_features[i[1]-box_corner[0], i[0]-box_corner[1]])])

								if joint_features['color_histograms'][ii] is None:
									joint_features['color_histograms'][ii] = color_histograms[i[1]-box_corner[0], i[0]-box_corner[1]]
								else:
									joint_features['color_histograms'][ii] = np.vstack([joint_features['color_histograms'][ii], deepcopy(color_histograms[i[1]-box_corner[0], i[0]-box_corner[1]])])

								if joint_features['lbp'][ii] is None:
									joint_features['lbp'][ii] = lbp_histograms[i[1]-box_corner[0], i[0]-box_corner[1]]
								else:
									joint_features['lbp'][ii] = np.vstack([joint_features['lbp'][ii], deepcopy(lbp_histograms[i[1]-box_corner[0], i[0]-box_corner[1]])])

							except:
								print "error"

				''' ---- Per Joint Classification ---- '''
				if not learn:
					try:
						# Geodesic clasification
						tmp = geodesic_features.reshape([-1, 6])
						tmp = np.array([x/x[-1] for x in tmp])
						tmp = np.nan_to_num(tmp)
						geo_clf_map = clf_geo.predict(tmp).reshape(im_pos.shape[:2])*mask[box]
						geo_clf_labels = geo_clf_map[pose_truth_uv[[0,1,4,7,10,13],1]-box_corner[0], pose_truth_uv[[0,1,4,7,10,13],0]-box_corner[1]]
						geo_accuracy += [geo_clf_labels == [0,1,4,7,10,13]]
						print 'G',np.mean(geo_accuracy,0), geo_clf_labels==[0,1,4,7,10,13]
						cv2.imshow('Geo', geo_clf_map/float(geo_clf_map.max()))
					except:
						pass

					try:
						# Color histogram classification
						color_test = color_approx.transform(color_histograms.reshape([-1, 5]))
						color_clf_map = clf_color.predict(color_test).reshape(im_pos.shape[:2])*mask[box]
						color_clf_labels = color_clf_map[pose_truth_uv[[0,1,4,7,10,13],1]-box_corner[0], pose_truth_uv[[0,1,4,7,10,13],0]-box_corner[1]]
						color_accuracy += [color_clf_labels == [0,1,4,7,10,13]]
						print 'C',np.mean(color_accuracy,0), color_clf_labels==[0,1,4,7,10,13]
						cv2.imshow('Col', color_clf_map/float(color_clf_map.max()))
					except:
						pass

					try:
						# lbp histogram classification
						lbp_test = color_approx.transform(lbp_histograms.reshape([-1, 10]))
						lbp_clf_map = clf_lbp.predict(lbp_test).reshape(im_pos.shape[:2])*mask[box]
						lbp_clf_labels = lbp_clf_map[pose_truth_uv[[0,1,4,7,10,13],1]-box_corner[0], pose_truth_uv[[0,1,4,7,10,13],0]-box_corner[1]]
						lbp_accuracy += [lbp_clf_labels == [0,1,4,7,10,13]]
						print 'L',np.mean(lbp_accuracy,0), lbp_clf_labels==[0,1,4,7,10,13]
						cv2.imshow('LBP', lbp_clf_map/float(lbp_clf_map.max()))
					except:
						pass

				pose_uv = pose_truth_uv
				pose = pose_truth

			# ''' ---- Accuracy ---- '''
			if 1 and not learn:
				# pose_truth = cam.users[0]
				error = pose_truth - pose
				# print "Error", error
				error_l2 = np.sqrt(np.sum(error**2, 1))
				# error_l2 = np.sqrt(np.sum(error[:,:2]**2, 1))
				joint_accuracy_all += [error_l2]
				accuracy = np.sum(error_l2 < 150) / 14.
				accuracy_all += [accuracy]
				print "Current", accuracy
				# print "Running avg:", np.mean(accuracy_all)
				# print "Joint avg (per-joint):", np.mean(joint_accuracy_all, -1)
				# print "Joint avg (overall):", np.mean(joint_accuracy_all)

			''' --- Visualization --- '''
			cam.colorIm = display_skeletons(cam.colorIm, pose_truth_uv, skel_type='Kinect', color=(0,255,0))
			cam.colorIm = display_skeletons(cam.colorIm, pose_uv, skel_type='Kinect')
			cam.visualize()

			# print "Extrema:", geo_clf_map[geodesic_markers[:,0], geodesic_markers[:,1]]
			# print "Skin:", geo_clf_map[hand_markers[:,0], hand_markers[:,1]]
			# print "Skin val:", hand_detector.skin_match[hand_markers[:,0], hand_markers[:,1]]
			# hand_data += [[x[0] for x in hand_markers],
							# [x[1] for x in hand_markers],
							# list(hand_detector.skin_match[hand_markers[:,0], hand_markers[:,1]])]

			# ------------------------------------------------------------

			# video_writer.write((geo_clf_map/float(geo_clf_map.max())*255.).astype(np.uint8))
			# video_writer.write(cam.colorIm[:,:,[2,1,0]])

			frame_count += frame_rate
	except:
		pass


	print "-- Results for subject {:d} action {:d}".format(subjects[0],actions[0])
	print "Running avg:", np.mean(accuracy_all)
	print "Joint avg (overall):", np.mean(joint_accuracy_all)
	# print 'Done'
	if learn:
		pose_database.save()
		print 'Pose database saved'

	embed()
	return






	''' --- Format Geodesic features ---'''
	geodesics_train = []
	geodesics_labels = []
	for i in xrange(len(joint_features['geodesic'])):
		# joint_features['geodesic'][i] = np.array([np.sort(x) for x in joint_features['geodesic'][i] if x[0] != 0])
		joint_features['geodesic'][i] = np.array([x/x.max() for x in joint_features['geodesic'][i] if x[0] != 0])
		ii = i
		if i not in [0,1,4,7,10,13]:
			ii=1
		else:
			geodesics_labels += [i*np.ones(len(joint_features['geodesic'][i]))]
	geodesics_train = np.vstack([joint_features['geodesic'][x] for x in [0,1,4,7,10,13]])
	# geodesics_train = np.vstack(joint_features['geodesic'])
	geodesics_labels = np.hstack(geodesics_labels)

	figure(1)
	title('Distances of each joint to first 6 geodesic extrema')
	for i in range(14):
		subplot(4,4,i+1)
		ylabel('Distance')
		xlabel('Sample')
		plot(joint_features['geodesic'][i])
		axis([0,400,0,1600])

	# Learn geodesic classifier
	clf_geo = SGDClassifier(n_iter=10000, alpha=.01, n_jobs=-1, class_weight='auto')
	clf_geo.fit(geodesics_train, geodesics_labels)
	print clf_geo.score(geodesics_train, geodesics_labels)
	geodesic_features = np.sort(geodesic_features)
	sgd_map = clf_geo.predict(geodesic_features.reshape([-1, 6])).reshape(im_pos.shape[:2])

	pickle.dump(clf_geo, open('geodesic_svm_sorted_scaled_5class.pkl','w'), pickle.HIGHEST_PROTOCOL)
	# clf_geo = pickle.load(open('geodesic_svm_sorted_scaled_5class.pkl'))

	''' --- Color Histogram features ---'''
	color_train = []
	color_labels = []
	for i in xrange(len(joint_features['color_histograms'])):
		ii = i
		if i not in [0,1,4,7,10,13]:
			ii=1
		else:
			color_labels += [i*np.ones(len(joint_features['color_histograms'][i]))]
		# color_labels += [i*np.ones(len(joint_features['color_histograms'][i]))]
	# color_train = np.vstack(joint_features['color_histograms'])
	color_train = np.vstack([joint_features['color_histograms'][x] for x in [0,1,4,7,10,13]])
	color_labels = np.hstack(color_labels)

	color_approx = AdditiveChi2Sampler()
	color_approx_train = color_approx.fit_transform(color_train)
	clf = SGDClassifier(n_iter=10000, alpha=.01, n_jobs=-1, class_weight='auto')
	clf.fit(color_approx_train, color_labels)
	print clf.score(color_approx_train, color_labels)
	color_test = color_approx.transform(color_histograms.reshape([-1, 5]))
	sgd_map = clf.predict(color_test).reshape(im_pos.shape[:2])*mask[box]

	figure(1)
	title('Color Histograms per Joint')
	for i in range(14):
		subplot(4,4,i+1)
		ylabel('Count')
		xlabel('Sample')
		plot(joint_features['color_histograms'][i])
		axis([0,10,0,30])

	for i in range(5):
		subplot(1,5,i+1)
		imshow(color_histograms[:,:,i])

	pickle.dump([clf,color_approx], open('color_histogram_approx_svm_5class.pkl','w'), pickle.HIGHEST_PROTOCOL)
	# clf_color,color_approx = pickle.load(open('color_histogram_approx_svm_5class.pkl'))

	''' --- LBP Histogram features ---'''
	color_train = []
	color_labels = []
	for i in xrange(len(joint_features['lbp'])):
		ii = i
		if i not in [0,1,4,7,10,13]:
			ii=1
		else:
			color_labels += [i*np.ones(len(joint_features['lbp'][i]))]
		# color_labels += [i*np.ones(len(joint_features['color_histograms'][i]))]
	# color_train = np.vstack(joint_features['color_histograms'])
	color_train = np.vstack([joint_features['lbp'][x] for x in [0,1,4,7,10,13]])
	color_labels = np.hstack(color_labels)

	color_approx = AdditiveChi2Sampler()
	color_approx_train = color_approx.fit_transform(color_train)
	clf = SGDClassifier(n_iter=10000, alpha=.01, n_jobs=-1, class_weight='auto')
	clf.fit(color_approx_train, color_labels)
	print clf.score(color_approx_train, color_labels)
	color_test = color_approx.transform(lbp_histograms.reshape([-1, 10]))
	sgd_map = clf.predict(color_test).reshape(im_pos.shape[:2])*mask[box]

	figure(1)
	title('LBP Histograms per Joint')
	for i in range(14):
		subplot(4,4,i+1)
		ylabel('Count')
		xlabel('Sample')
		plot(joint_features['lbp'][i])
		axis([0,10,0,30])

	for i in range(5):
		subplot(1,5,i+1)
		imshow(color_histograms[:,:,i])

	pickle.dump([clf,color_approx], open('lbp_histogram_approx_svm_5class.pkl','w'), pickle.HIGHEST_PROTOCOL)
	# clf_lbp,lbp_approx = pickle.load(open('lbp_histogram_approx_svm_5class.pkl'))




if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	parser.add_option('-l', '--learn', dest='learn', action="store_true", default=False, help='Training phase')
	parser.add_option('-a', '--actions', dest='actions', type='int', action='append', default=[], help='Training phase')
	parser.add_option('-s', '--subjects', dest='subjects', type='int', action='append', default=[], help='Training phase')
	(opt, args) = parser.parse_args()

	main(visualize=opt.viz, learn=opt.learn, actions=opt.actions, subjects=opt.subjects)



