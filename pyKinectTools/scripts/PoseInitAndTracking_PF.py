"""
Main file for training multi-camera pose
"""

#import os
import time
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
	learn = True
	learn = False
	if actions is []:
		actions = [2]
	if subjects is []:
		subjects = [2]
	# actions = [1]
	# actions = [1, 2, 3, 4, 5]
	# subjects = [1]
	if 0:
		MHAD = True
		cam = MHADPlayer(base_dir='/Users/colin/Data/BerkeleyMHAD/', kinect=1, actions=actions, subjects=subjects, reps=[1], get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
	else:
		MHAD = False
		cam = KinectPlayer(base_dir='./', device=1, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
		bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/CIRL_Background_A.tif')
		# bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/Wall_Background_A.tif')
		# bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/Office_Background_A.tif')
		# cam = KinectPlayer(base_dir='./', device=2, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
		# bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/CIRL_Background_B.tif')
		cam.bgSubtraction.backgroundModel = np.array(bg.getdata()).reshape([240,320]).clip(0, 4500)
	height, width = cam.depthIm.shape
	skel_previous = None

	face_detector = FaceDetector()
	hand_detector = HandDetector(cam.depthIm.shape)
	# curve_detector = CurveDetector(cam.depthIm.shape)

	# Video writer
	video_writer = cv2.VideoWriter("/Users/colin/Desktop/test.avi", cv2.cv.CV_FOURCC('M','J','P','G'), 15, (640,480))
	# video_writer = cv2.VideoWriter("/Users/colin/Desktop/test.avi", cv2.cv.CV_FOURCC('M','J','P','G'), 15, (320,240))

	# Save Background model
	# im = Image.fromarray(cam.depthIm.astype(np.int32), 'I')
	# im.save("/Users/Colin/Desktop/k2.png")

	# Setup pose database
	append = True
	append = False
	# pose_database = PoseDatabase("PoseDatabase.pkl", learn=learn, search_joints=[0,4,7,10,13], append=append)
	pose_database = PoseDatabase("PoseDatabase.pkl", learn=learn, search_joints=[0,2,4,5,7,10,13], append=append, scale=1.2)
	pose_prob = np.ones(len(pose_database.database), dtype=np.float)/len(pose_database.database)
	# # Visualize database
	# for i,p in enumerate(np.array(pose_database.database)[:200*10:200]):
	# 	im = np.zeros([480,640], np.uint8)+255
	# 	print i
	# 	subplot(1,10,i+1)
	# 	p_tmp = cam.camera_model.world2im(p+[0,0,2500])
	# 	im = display_skeletons(im, p_tmp, skel_type='Kinect', color=(0,0,0))
	# 	imshow(im[:,150:450])
	# 	axis('off')
	# 	gray()


	# Setup Tracking
	skel_init, joint_size, constraint_links, features_joints,skel_parts, convert_to_kinect = get_14_joint_properties()
	constraint_values = []
	for c in constraint_links:
		constraint_values += [np.linalg.norm(skel_init[c[0]]-skel_init[c[1]], 2)]
	constraint_values = np.array(constraint_values)

	skel_current = None#skel_init.copy()
	skel_previous = None#skel_current.copy()
	skel_previous_uv = None

	# Evaluation
	accuracy_all_db = []
	accuracy_all_track = []
	joint_accuracy_db = []
	joint_accuracy_track = []
	# geo_accuracy = []
	# color_accuracy = []
	# lbp_accuracy = []

	frame_count = 0
	frame_rate = 2
	if not MHAD:
		# cam.next(350)
		# cam.next(600)
		# cam.next(700)
		pass
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
			mask_box = mask[box]

			''' ---------- ----------------------------------- --------'''
			''' ---------- ----------------------------------- --------'''

			''' ---- Calculate Detectors ---- '''
			# Face detection
			face_detector.run(im_color[box])
			# Skin detection
			hand_markers = hand_detector.run(im_color[box], n_peaks=3)
			# Calculate Geodesic Extrema
			im_pos = cam.camera_model.im2PosIm(cam.depthIm*mask)[box] * mask_box[:,:,None]
			geodesic_markers = geodesic_extrema_MPI(im_pos, iterations=5, visualize=False)
			_, geo_map = geodesic_extrema_MPI(im_pos, iterations=1, visualize=True)
			geodesic_markers_pos = im_pos[geodesic_markers[:,0], geodesic_markers[:,1]]

			markers = list(geodesic_markers) + list(hand_markers) #+ list(lop_markers) + curve_markers
			markers = np.array([list(x) for x in markers])
			if np.any(markers==0):
				continue

			''' ---- Database lookup ---- '''
			time_t0 = time.time()
			pts_mean = im_pos[(im_pos!=0)[:,:,2]].mean(0)
			if learn:
				# Normalize pose
				pose_uv = pose_truth_uv

				if np.any(pose_uv==0):
					print "skip"
					# print pose_uv
					frame_count += frame_rate
					continue
				else:
					print 'yay'

				# Markers can be just outside of bounds
				markers = list(geodesic_markers) + hand_markers
				markers = np.array([list(x) for x in markers])
				pose_database.update(pose_truth-pts_mean, keys=im_pos[markers[:,0],markers[:,1]]-pts_mean)
				continue
			else:
				# Concatenate markers
				markers = list(geodesic_markers) + hand_markers
				markers = np.array([list(x) for x in markers])

				# Normalize pose
				pts = im_pos[markers[:,0], markers[:,1]]
				pts = np.array([x for x in pts if x[0] != 0])
				pts -= pts_mean

				# Get closest pose
				# Based on markers/raw positions
				poses_obs, pose_error = pose_database.query(pts, knn=1, return_error=True)

				# Based on markers/keys:
				# pts = im_pos[markers[:,0], markers[:,1]] - pts_mean
				# # poses, pose_error = pose_database.query_tree(pts, knn=len(pose_database.database), return_error=True)
				# # poses, pose_error = pose_database.query_flann(pts, knn=len(pose_database.database), return_error=True)
				# pose_error = np.sqrt(np.sum((pose_database.keys - pts.reshape([27]))**2, 1))

				observation_variance = 100.
				prob_obervation = np.exp(-pose_error / observation_variance) / np.sum(np.exp(-pose_error/observation_variance))


				# subplot(2,2,1)
				# plot(prob_obervation)
				# subplot(2,2,2)
				# plot(prob_motion)
				# subplot(2,2,3)
				# plot(pose_prob_new)
				# subplot(2,2,4)
				# plot(pose_prob)
				# show()

				# inference = 'NN'
				inference = 'Bayes'
				# inference = 'PF'
				if inference=='NN': # Nearest neighbor
					poses = [poses_obs[0]]
					# print poses_obs

				elif inference=='Bayes': # Bayes
					if frame_count is 0:
						skel_previous = poses_obs[0].copy()
					poses_m, pose_m_error = pose_database.query(skel_previous-pts_mean+(np.random.random([3,14])-.5).T*30, knn=5, return_error=True)
					motion_variance = 10000.
					prob_motion = np.exp(-pose_m_error / motion_variance) / np.sum(np.exp(-pose_m_error/motion_variance))
					pose_prob_new = prob_obervation*prob_motion
					pose_prob = (pose_prob_new+pose_prob).T/2.
					prob_sorted = np.argsort(pose_prob)

					poses = pose_database.database[prob_sorted[-1:]]

				# Particle Filter
				elif inference=='PF':
					prob_sorted = np.argsort(pose_prob)
					poses = pose_database.database[prob_sorted[-5:]]


				# im_pos -= pts_mean
				# R,t = IterativeClosestPoint(pose, im_pos.reshape([-1,3])-pts_mean, max_iters=5, min_change=.001, pt_tolerance=10000)
				# pose = np.dot(R.T, pose.T).T - t
				# pose = np.dot(R, pose.T).T + t

				# scale = 1.
				# poses *= scale

				poses += pts_mean
			# print "DB time:", time.time() - time_t0
			# embed()
			''' ---- Tracker ---- '''
			surface_map = nd.distance_transform_edt(-nd.binary_erosion(mask_box), return_distances=False, return_indices=True)

			if skel_previous_uv is None:
				skel_previous = poses[0].copy()
				skel_current = poses[0].copy()
				pose_tmp = cam.camera_model.world2im(poses[0], cam.depthIm.shape)
				skel_previous_uv = pose_tmp.copy()
				skel_current_uv = pose_tmp.copy()

			pose_weights = np.zeros(len(poses), dtype=np.float)
			pose_updates = []
			pose_updates_uv = []

			time_t0 = time.time()
			# 2) Sample poses
			if inference in ['PF', 'Bayes']:
				for pose_i, pose in enumerate(poses):

					skel_current = skel_previous.copy()
					skel_current_uv = skel_previous_uv.copy()

					pose_uv = cam.camera_model.world2im(pose, cam.depthIm.shape)
					try:
						pose_uv[:,:2] = surface_map[:, pose_uv[:,0]-box_corner[0], pose_uv[:,1]-box_corner[1]].T + [box_corner[0], box_corner[1]]
					except:
						pass
					pose = cam.camera_model.im2world(pose_uv, cam.depthIm.shape)

					# ---- (Step 2) Update pose state, x ----
					correspondence_displacement = skel_previous - pose
					lambda_p = .0
					lambda_c = 1.
					skel_prev_difference = (skel_current - skel_previous)
					# print skel_prev_difference
					skel_current = skel_previous \
									+ lambda_p  * skel_prev_difference \
									- lambda_c  * correspondence_displacement#\

					# ---- (Step 3) Add constraints ----
					# A: Link lengths / geometry
					# skel_current = link_length_constraints(skel_current, constraint_links, constraint_values, alpha=.5)
					# skel_current = geometry_constraints(skel_current, joint_size, alpha=0.5)
					# skel_current = collision_constraints(skel_current, constraint_links)

					skel_current_uv = (cam.camera_model.world2im(skel_current, cam.depthIm.shape) - [box[0].start, box[1].start, 0])#/mask_interval
					skel_current_uv = skel_current_uv.clip([0,0,0], [box[0].stop-box[0].start-1, box[1].stop-box[1].start-1, 9999])
					# B: Ray-cast constraints
					skel_current, skel_current_uv = ray_cast_constraints(skel_current, skel_current_uv, im_pos, surface_map, joint_size)

					# Map back from mask to image
					# try:
						# skel_current_uv[:,:2] = surface_map[:, skel_current_uv[:,0], skel_current_uv[:,1]].T# + [box_corner[0], box_corner[1]]
					# except:
						# pass

					# ---- (Step 4) Update the confidence ----
					if inference=='PF':
						time_t1 = time.time()
						## Calc distance between each pixel and all joints
						px_corr = np.zeros([im_pos.shape[0], im_pos.shape[1], 14])
						for i,s in enumerate(skel_current):
							px_corr[:,:,i] = np.sqrt(np.sum((im_pos - s)**2, -1))# / joint_size[i]**2

						# for i,s in enumerate(pose_uv):
						# for i,s in enumerate(skel_current_uv):
						# 	''' Problem: need to constrain pose_uv to mask '''
							# _, geo_map = geodesic_extrema_MPI(im_pos, [s[0],s[1]], iterations=1, visualize=True)
							# px_corr[:,:,i] = geo_map
							# subplot(2,7,i+1)
							# imshow(geo_map, vmin=0, vmax=2000)
							# axis('off')
							# px_corr[geo_map==0,i] = 9999

						px_label = np.argmin(px_corr, -1)*mask_box
						px_label_flat = px_label[mask_box].flatten()

						cv2.imshow('gMap', (px_corr.argmin(-1)+1)/15.*mask_box)
						cv2.waitKey(1)

						# Project distance to joint's radius
						px_joint_displacement = im_pos[mask_box] - skel_current[px_label_flat]
						px_joint_magnitude = np.sqrt(np.sum(px_joint_displacement**2,-1))
						joint_mesh_pos = skel_current[px_label_flat] + px_joint_displacement*(joint_size[px_label_flat]/px_joint_magnitude)[:,None]
						px_joint_displacement = joint_mesh_pos - im_pos[mask_box]
						# Ensure pts aren't too far away (these are noise!)
						px_joint_displacement[np.abs(px_joint_displacement) > 500] = 0

						if 0:
							x = im_pos.copy()*0
							x[mask_box] = joint_mesh_pos

							for i in range(3):
								subplot(1,4,i+1)
								imshow(x[:,:,i])
								axis('off')
							subplot(1,4,4)
							imshow((px_label+1)*mask_box)

						# Calc the correspondance change in position for each joint
						correspondence_displacement = np.zeros([len(skel_current), 3])
						ii = 0
						for i,_ in enumerate(skel_current):
							labels = px_label_flat==i
							correspondence_displacement[i] = np.sum(px_joint_displacement[px_label_flat==ii], 0) / np.sum(px_joint_displacement[px_label_flat==ii]!=0)
							ii+=1
						correspondence_displacement = np.nan_to_num(correspondence_displacement)


					# print "time:", time.time() - time_t1
					# Likelihood
					motion_variance = 500
					prob_motion = np.exp(-np.mean(np.sum((pose-skel_previous)**2,1)/motion_variance**2))

					if inference == 'PF':

						correspondence_variance = 40
						prob_coor 	= np.exp(-np.mean(np.sum(correspondence_displacement**2,1)/correspondence_variance**2))
						prob = prob_motion * prob_coor
					prob = prob_motion

					# Viz correspondences
					# x = im_pos.copy()*0
					# x[mask_box] = px_joint_displacement

					# for i in range(3):
					# 	subplot(1,4,i+1)
					# 	imshow(x[:,:,i])
					# 	axis('off')
					# subplot(1,4,4)
					# imshow((px_label+1)*mask_box)
					# # embed()
					# # for j in range(3):
					# # 	for i in range(14):
					# # 		subplot(3,14,j*14+i+1)
					# # 		imshow(x[:,:,j]*((px_label==i)*mask_box))
					# # 		axis('off')
					# show()

					# prob = link_length_probability(skel_current, constraint_links, constraint_values, 100)

					# print frame_count
					# print "Prob:", np.mean(prob)#, np.min(prob), prob
					# thresh = .05
					# if np.min(prob) < thresh:
					# 	# print 'Resetting pose'
					# 	for c in constraint_links[prob<thresh]:
					# 		for cc in c:
					# 			skel_current_uv[c] = pose_uv[c] - [box[0].start, box[1].start, 0]
					# 			skel_current[c] = pose[c]
						# skel_current_uv = pose_uv.copy() - [box[0].start, box[1].start, 0]
						# skel_current = pose.copy()

					skel_current_uv = skel_current_uv + [box[0].start, box[1].start, 0]
					skel_current = cam.camera_model.im2world(skel_current_uv, cam.depthIm.shape)
					# print 'Error:', np.sqrt(np.sum((pose_truth-skel_current)**2, 0))
					pose_weights[pose_i] = prob
					# pose_updates += [skel_current.copy()]
					# pose_updates_uv += [skel_current_uv.copy()]
					pose_updates += [pose.copy()]
					pose_updates_uv += [pose_uv.copy()]

					cam.colorIm = display_skeletons(cam.colorIm, skel_current_uv, skel_type='Kinect', color=(0,0,pose_i*40+50))
					# cam.colorIm = display_skeletons(cam.colorIm, pose_uv, skel_type='Kinect', color=(0,pose_i*40+50,pose_i*40+50))
				# print "Tracking time:", time.time() - time_t0
				# Update for next round

				pose_ind = np.argmax(pose_weights)
				# print "Pickled:", pose_ind
				skel_previous = pose_updates[pose_ind].copy()
				skel_previous_uv = pose_updates_uv[pose_ind].copy()
				# print pose_weights
			else:
				pose = poses[0]
				skel_previous = pose.copy()
				pose_uv = cam.camera_model.world2im(skel_previous, cam.depthIm.shape)
				skel_current_uv = pose_uv.copy()
				skel_previous_uv = pose_uv.copy()


			''' ---- Accuracy ---- '''
			# embed()
			if not learn:
				# pose_truth = cam.users[0]
				error_db = pose_truth - pose
				error_track = pose_truth - skel_previous
				# print "Error", error
				error_l2_db = np.sqrt(np.sum(error_db**2, 1))
				error_l2_track = np.sqrt(np.sum(error_track**2, 1))
				joint_accuracy_db += [error_l2_db]
				joint_accuracy_track += [error_l2_track]
				accuracy_db = np.sum(error_l2_db < 150) / 14.
				accuracy_track = np.sum(error_l2_track < 150) / 14.
				# print "Current db:", accuracy_db, error_l2_db.mean()
				# print "Current track:", accuracy_track, error_l2_track.mean()
				# print "Running avg (track):", np.mean(accuracy_all_track)
				# print "Joint avg (overall track):", np.mean(joint_accuracy_track)
				# print ""
				accuracy_all_db += [accuracy_db]
				accuracy_all_track += [accuracy_track]
				# print "Running avg:", np.mean(accuracy_all)
				# print "Joint avg (per-joint):", np.mean(joint_accuracy_all, -1)
				# print "Joint avg (overall):", np.mean(joint_accuracy_all)

			''' --- Visualization --- '''

			display_markers(cam.colorIm, hand_markers[:2], box, color=(0,250,0))
			if len(hand_markers) > 2:
				display_markers(cam.colorIm, [hand_markers[2]], box, color=(0,200,0))
			display_markers(cam.colorIm, geodesic_markers, box, color=(200,0,0))
			# display_markers(cam.colorIm, curve_markers, box, color=(0,100,100))
			# display_markers(cam.colorIm, lop_markers, box, color=(0,0,200))

			cam.colorIm = display_skeletons(cam.colorIm, pose_truth_uv, skel_type='Kinect', color=(0,255,0))
			# cam.colorIm = display_skeletons(cam.colorIm, pose_uv, skel_type='Kinect')
			cam.colorIm = display_skeletons(cam.colorIm, skel_current_uv, skel_type='Kinect', color=(255,0,0))
			cam.visualize(color=True, depth=False)
			# cam.visualize(color=True, depth=True)

			# ------------------------------------------------------------

			# video_writer.write((geo_clf_map/float(geo_clf_map.max())*255.).astype(np.uint8))
			# video_writer.write(cam.colorIm[:,:,[2,1,0]])

			frame_count += frame_rate
	except:
		pass


	print "-- Results for subject {:d} action {:d}".format(subjects[0],actions[0])
	print "Running avg (db):", np.mean(accuracy_all_db)
	print "Running avg (track):", np.mean(accuracy_all_track)
	print "Joint avg (overall db):", np.mean(joint_accuracy_db)
	print "Joint avg (overall track):", np.mean(joint_accuracy_track)
	# print 'Done'

	embed()
	if learn:
		pose_database.save()




if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	parser.add_option('-l', '--learn', dest='learn', action="store_true", default=False, help='Training phase')
	parser.add_option('-a', '--actions', dest='actions', type='int', action='append', default=[], help='Training phase')
	parser.add_option('-s', '--subjects', dest='subjects', type='int', action='append', default=[], help='Training phase')
	(opt, args) = parser.parse_args()

	main(visualize=opt.viz, learn=opt.learn, actions=opt.actions, subjects=opt.subjects)



