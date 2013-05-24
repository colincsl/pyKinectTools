"""
Main file for training multi-camera pose
"""

import sys
import time
import traceback
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

from RGBDActionDatasets.dataset_readers.KinectPlayer import KinectPlayer, display_help
from RGBDActionDatasets.dataset_readers.RealtimePlayer import RealtimePlayer
# from pyKinectTools.dataset_readers.KinectPlayer import KinectPlayer, display_help
# from pyKinectTools.dataset_readers.RealtimePlayer import RealtimePlayer
from RGBDActionDatasets.dataset_readers.MHADPlayer import MHADPlayer
# from pyKinectTools.dataset_readers.MHADPlayer import MHADPlayer
from pyKinectTools.utils.DepthUtils import *
from pyKinectTools.utils.SkeletonUtils import display_skeletons, transform_skels, kinect_to_msr_skel, msr_to_kinect_skel
from pyKinectTools.algs.GeodesicSkeleton import *
from pyKinectTools.algs.PoseTracking import *

from IPython import embed
np.seterr(all='ignore')

# -------------------------MAIN------------------------------------------

def main(visualize=False, learn=False, actions=None, subjects=None, n_frames=220):

	# search_joints=[0,2,4,5,7,10,13]
	search_joints=range(14)
	interactive = True
	interactive = False
	save_results = False
	if 1:
		learn = False
	else:
		learn = True
		actions = [1, 2, 3, 4, 5]
		subjects = [5]

	if 0:
		dataset = 'MHAD'
		cam = MHADPlayer(base_dir='/Users/colin/Data/BerkeleyMHAD/', kinect=1, actions=actions, subjects=subjects, reps=[1], get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
	elif 1:
		dataset = 'JHU'
		cam = KinectPlayer(base_dir='./', device=1, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
		bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/CIRL_Background_A.tif')
		# bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/Wall_Background_A.tif')
		# bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/Office_Background_A.tif')
		# bg = Image.open('/Users/colin/Data/WICU_May2013_C2/WICU_C2_Background.tif')
		# cam = KinectPlayer(base_dir='./', device=2, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
		# bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/CIRL_Background_B.tif')
		cam.bgSubtraction.backgroundModel = np.array(bg.getdata()).reshape([240,320]).clip(0, 4500) - 000.
	else:
		# Realtime
		dataset = 'RT'
		cam = RealtimePlayer(device=0, edit=True, get_depth=True, get_color=True, get_skeleton=True)
		# cam.set_bg_model('box', 2500)
		tmp = cam.depthIm
		tmp[tmp>4000] = 4000
		cam.set_bg_model(bg_type='static', param=tmp)


	height, width = cam.depthIm.shape
	skel_previous = None

	face_detector = FaceDetector()
	hand_detector = HandDetector(cam.depthIm.shape)

	# Video writer
	video_writer = cv2.VideoWriter("/Users/colin/Desktop/test.avi", cv2.cv.CV_FOURCC('M','J','P','G'), 15, (640,480))

	# Save Background model
	# im = Image.fromarray(cam.depthIm.astype(np.int32), 'I')
	# im.save("/Users/Colin/Desktop/k2.png")

	# Setup pose database
	append = True
	# append = False
	# pose_database = PoseDatabase("PoseDatabase.pkl", learn=learn, search_joints=[0,4,7,10,13], append=append)
	pose_database = PoseDatabase("PoseDatabase.pkl", learn=learn, search_joints=search_joints,
									append=append, scale=1.1, n_clusters=-1)#1000
	pose_prob = np.ones(len(pose_database.database), dtype=np.float)/len(pose_database.database)


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
	if not learn:
		try:
			results = pickle.load(open('Accuracy_Results.pkl'))
		except:
			results = { 'subject':[], 		'action':[],		'accuracy_all':[],
						'accuracy_mean':[],	'joints_all':[],
						'joint_mean':[],	'joint_median':[]}

	frame_count = 0
	frame_rate = 1
	if dataset == 'JHU':
		cam.next(350)
		# cam.next(700)
		pass
	frame_prev = 0
	try:
	# if 1:
		while cam.next(frame_rate):# and frame_count < n_frames:
			# Print every once in a while
			if frame_count - frame_prev > 99:
				print ""
				print "Frame #{0:d}".format(frame_count)
				frame_prev = frame_count

			if dataset in ['MHAD', 'JHU']:
				users = deepcopy(cam.users)
			else:
				users = deepcopy(cam.user_skels)

			ground_truth = False
			if dataset in ['RT','JHU']:
				if len(users) > 0:
					if not np.any(users[0][0] == -1):
						ground_truth = True
						users[0][:,1] *= -1
						cam.users_uv_msr = [cam.camera_model.world2im(users[0], cam.depthIm.shape)]
			else:
				ground_truth = True

			# Apply mask to image
			mask = cam.get_person(200) == 1 # > 0
			# cv2.imshow('bg',(mask*255).astype(np.uint8))
			cv2.imshow('bg',cam.colorIm)
			# cv2.waitKey(1)
			if type(mask)==bool or np.all(mask==False):
				# print "No mask"
				continue
			# cv2.imshow('bg',cam.bgSubtraction.backgroundModel)
			cv2.imshow('bg',(mask*255).astype(np.uint8))

			im_depth =  cam.depthIm
			# if dataset in ['RT']:
				# cam.depthIm[cam.depthIm>2500] = 0
			im_color = cam.colorIm*mask[:,:,None]
			cam.colorIm *= mask[:,:,None]
			if ground_truth:
				pose_truth = users[0]
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
			# face_detector.run(im_color[box])
			# Skin detection
			# hand_markers = hand_detector.run(im_color[box], n_peaks=3)
			hand_markers = []
			# Calculate Geodesic Extrema
			im_pos = cam.camera_model.im2PosIm(cam.depthIm*mask)[box]
			# geodesic_markers = geodesic_extrema_MPI(im_pos, iterations=5, visualize=False)
			geodesic_markers = geodesic_extrema_MPI(im_pos, iterations=10, visualize=False)
			if len(geodesic_markers) == 0:
				print "No markers"
				continue

			# Concatenate markers
			markers = list(geodesic_markers) + list(hand_markers) #+ list(lop_markers) + curve_markers
			markers = np.array([list(x) for x in markers])
			if np.any(markers==0):
				print "Bad markers"
				continue

			''' ---- Database lookup ---- '''
			time_t0 = time.time()
			pts_mean = im_pos[(im_pos!=0)[:,:,2]].mean(0)
			if learn and ground_truth:
				# pose_uv = pose_truth_uv
				if np.any(pose_truth_uv==0):
					frame_count += frame_rate
					if not interactive:
						continue

				# Markers can be just outside of bounds
				markers = list(geodesic_markers) + hand_markers
				markers = np.array([list(x) for x in markers])
				# pose_database.update(pose_truth-pts_mean, keys=im_pos[markers[:,0],markers[:,1]]-pts_mean)
				pose_database.update(pose_truth-pts_mean)
				if not interactive:
					continue
			# else:
			if 1:
				# Normalize pose
				pts = im_pos[markers[:,0], markers[:,1]]
				pts = np.array([x for x in pts if x[0] != 0])
				pts -= pts_mean

				# Get closest pose
				# Based on markers/raw positions
				# poses_obs, pose_error = pose_database.query(pts, knn=1, return_error=True)
				pose_error = pose_query(pts, np.array(pose_database.database), search_joints=search_joints)
				# pose_error = query_error(pts, pose_database.trees, search_joints=search_joints)

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
					poses_obs, _ = pose_database.query(pts, knn=1, return_error=True)
					poses = [poses_obs[0]]

				elif inference=='Bayes': # Bayes
					if frame_count is 0:
						poses_obs, _ = pose_database.query(pts, knn=1, return_error=True)
						skel_previous = poses_obs[0].copy()
					# poses_m, pose_m_error = pose_database.query(skel_previous-pts_mean, knn=1, return_error=True)
					pose_m_error = pose_query(skel_previous-pts_mean, np.array(pose_database.database), search_joints=search_joints)
					# poses_m, pose_m_error = pose_database.query(skel_previous-pts_mean+(np.random.random([3,14])-.5).T*30, knn=5, return_error=True)
					motion_variance = 10000.
					prob_motion = np.exp(-pose_m_error / motion_variance) / np.sum(np.exp(-pose_m_error/motion_variance))
					pose_prob_new = prob_obervation*prob_motion
					if pose_prob_new.shape == pose_prob.shape:
						pose_prob = (pose_prob_new+pose_prob).T/2.
					else:
						pose_prob = pose_prob_new.T
					prob_sorted = np.argsort(pose_prob)
					poses = [pose_database.database[np.argmax(pose_prob)]]

					# poses = pose_database.database[prob_sorted[-1:]]

				# Particle Filter
				elif inference=='PF':
					prob_sorted = np.argsort(pose_prob)
					poses = pose_database.database[prob_sorted[-5:]]

				## ICP
				# im_pos -= pts_mean
				# R,t = IterativeClosestPoint(pose, im_pos.reshape([-1,3])-pts_mean, max_iters=5, min_change=.001, pt_tolerance=10000)
				# pose = np.dot(R.T, pose.T).T - t
				# pose = np.dot(R, pose.T).T + t

				# scale = 1.
				# poses *= scale

				poses += pts_mean
			# print "DB time:", time.time() - time_t0
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

						# cv2.imshow('gMap', (px_corr.argmin(-1)+1)/15.*mask_box)
						# cv2.waitKey(1)

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
			if ground_truth:
				error_track = pose_truth - skel_previous
				error_track *= np.any(pose_truth!=0, 1)[:,None]
				error_l2_track = np.sqrt(np.sum(error_track**2, 1))
				joint_accuracy_track += [error_l2_track]
				accuracy_track = np.sum(error_l2_track < 150) / 14.
				print "Current track:", accuracy_track, error_l2_track.mean()
				print "Running avg (track):", np.mean(accuracy_all_track)
				# print "Joint avg (overall track):", np.mean(joint_accuracy_track)
				print ""
				accuracy_all_track += [accuracy_track]

			''' --- Visualization --- '''

			display_markers(cam.colorIm, hand_markers[:2], box, color=(0,250,0))
			if len(hand_markers) > 2:
				display_markers(cam.colorIm, [hand_markers[2]], box, color=(0,200,0))
			display_markers(cam.colorIm, geodesic_markers, box, color=(200,0,0))
			# display_markers(cam.colorIm, curve_markers, box, color=(0,100,100))
			# display_markers(cam.colorIm, lop_markers, box, color=(0,0,200))

			if ground_truth:
				cam.colorIm = display_skeletons(cam.colorIm, pose_truth_uv, skel_type='Kinect', color=(0,255,0))
			cam.colorIm = display_skeletons(cam.colorIm, skel_current_uv, skel_type='Kinect', color=(255,0,0))
			cam.visualize(color=True, depth=False)

			# ------------------------------------------------------------

			# video_writer.write((geo_clf_map/float(geo_clf_map.max())*255.).astype(np.uint8))
			# video_writer.write(cam.colorIm[:,:,[2,1,0]])

			frame_count += frame_rate
	except:
		traceback.print_exc(file=sys.stdout)
		pass


	try:
		print "-- Results for subject {:d} action {:d}".format(subjects[0],actions[0])
	except:
		pass
	# print "Running avg (db):", np.mean(accuracy_all_db)
	print "Running mean (track):", np.mean(accuracy_all_track)
	# print "Joint avg (overall db):", np.mean(joint_accuracy_db)
	print "Joint mean (overall track):", np.mean(joint_accuracy_track)
	print "Joint median (overall track):", np.median(joint_accuracy_track)
	# print 'Done'

	embed()
	if learn:
		pose_database.save()
	elif save_results:
		# Save results:
		results['subject'] += [subjects[0]]
		results['action'] += [actions[0]]
		results['accuracy_all'] += [accuracy_all_track]
		results['accuracy_mean'] += [np.mean(accuracy_all_track)]
		results['joints_all'] += [joint_accuracy_track]
		results['joint_mean'] += [np.mean(joint_accuracy_track)]
		results['joint_median'] += [np.median(joint_accuracy_track)]
		pickle.dump(results, open('/Users/colin/Data/BerkeleyMHAD/Accuracy_Results.pkl', 'w'))




if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	parser.add_option('-l', '--learn', dest='learn', action="store_true", default=False, help='Training phase')
	parser.add_option('-a', '--actions', dest='actions', type='int', action='append', default=[], help='Training phase')
	parser.add_option('-s', '--subjects', dest='subjects', type='int', action='append', default=[], help='Training phase')
	(opt, args) = parser.parse_args()

	main(visualize=opt.viz, learn=opt.learn, actions=opt.actions, subjects=opt.subjects)



