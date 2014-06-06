"""
Main file for training multi-camera pose
"""

#import os
#import time
import itertools as it
import optparse
import cPickle as pickle

import numpy as np
import cv2
import scipy.misc as sm
import scipy.ndimage as nd
import Image
import skimage
from skimage import color
from skimage.draw import line, circle
from skimage.color import rgb2gray,gray2rgb, rgb2lab
from skimage.feature import hog, local_binary_pattern, match_template, peak_local_max

from pyKinectTools.dataset_readers.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.utils.DepthUtils import *
from pyKinectTools.utils.SkeletonUtils import display_skeletons, transform_skels, kinect_to_msr_skel, msr_to_kinect_skel
from pyKinectTools.dataset_readers.MSR_DailyActivities import MSRPlayer
from pyKinectTools.dataset_readers.MHADPlayer import MHADPlayer
from pyKinectTools.dataset_readers.EVALPlayer import EVALPlayer
from pyKinectTools.algs.GeodesicSkeleton import *
from pyKinectTools.algs.HistogramOfOpticalFlow import hog2image
from pyKinectTools.algs.BackgroundSubtraction import fill_image
from pyKinectTools.algs.PoseTracking import *
from pyKinectTools.algs.LocalOccupancyPattern import local_occupancy_pattern

from IPython import embed
np.seterr(all='ignore')

from joblib import Parallel, delayed


# -------------------------MAIN------------------------------------------

def main(visualize=False, learn=False, patch_size=32, n_frames=2500):

	if 1:
		get_color = True
		cam = MHADPlayer(base_dir='/Users/colin/Data/BerkeleyMHAD/', kinects=[1], actions=[1], subjects=[1], get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
		# cam = KinectPlayer(base_dir='./', device=2, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
		# cam.bgSubtraction.backgroundModel = sm.imread('/Users/colin/Data/CIRL_28Feb2013/depth/59/13/47/device_1/depth_59_13_47_4_13_35507.png').clip(0, 4500)
		# bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/Office_Background_B.tif')
		# bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/CIRL_Background_B.tif')
		# bg = Image.open('/Users/colin/Data/JHU_RGBD_Pose/Wall_Background_B.tif')
		# cam.bgSubtraction.backgroundModel = np.array(bg.getdata()).reshape([240,320]).clip(0, 4500)
		# embed()
		# cam = KinectPlayer(base_dir='./', device=2, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=True, fill_images=False)
	elif 0:
		get_color = False
		cam = EVALPlayer(base_dir='/Users/colin/Data/EVAL/', bg_subtraction=True, get_depth=True, get_skeleton=True, fill_images=False)
	elif 0:
		get_color = False
		cam = MSRPlayer(base_dir='/Users/colin/Data/MSR_DailyActivities/Data/', actions=[1], subjects=[1,2,3,4,5], bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=True, fill_images=False)

	embed()
	height, width = cam.depthIm.shape

	skel_names = np.array(['head', 'neck', 'torso', 'l_shoulder', 'l_elbow', 'l_hand', \
				'r_shoulder', 'r_elbow', 'r_hand', 'l_hip', 'l_knee', 'l_foot',\
				'r_hip', 'r_knee', 'r_foot'])

	# skel_init, joint_size, constraint_links, features_joints,convert_to_kinect = get_11_joint_properties()
	skel_init, joint_size, constraint_links, features_joints,skel_parts, convert_to_kinect = get_13_joint_properties()
	# skel_init, joint_size, constraint_links, features_joints,skel_parts, convert_to_kinect = get_14_joint_properties()
	# skel_init, joint_size, constraint_links, features_joints,convert_to_kinect = get_15_joint_properties()
	constraint_values = []
	for c in constraint_links:
		constraint_values += [np.linalg.norm(skel_init[c[0]]-skel_init[c[1]], 2)]
	constraint_values = np.array(constraint_values)

	skel_current = skel_init.copy()
	skel_previous = skel_current.copy()

	face_detector = FaceDetector()
	hand_template = sm.imread('/Users/colin/Desktop/fist.png')[:,:,2]
	hand_template = (255 - hand_template)/255.
	if height == 240:
		hand_template = cv2.resize(hand_template, (10,10))
	else:
		hand_template = cv2.resize(hand_template, (20,20))

	frame_count = 0
	if get_color and height==240:
		cam.next(220)

	accuracy_measurements = {'overall':[], 'per_joint':[]}

	# Video writer
	# print '1'
	video_writer = cv2.VideoWriter("/Users/colin/Desktop/test.avi", cv2.cv.CV_FOURCC('M','J','P','G'), 15, (320,240))
	# print '1'

	# embed()
	while cam.next(1) and frame_count < n_frames:
		print ""
		print "Frame #{0:d}".format(frame_count)
		# Get rid of bad skeletons
		if type(cam.users)==dict:
			cam_skels = [np.array(cam.users[s]['jointPositions'].values()) for s in cam.users]
		else:
			cam_skels = [np.array(s) for s in cam.users]
		cam_skels = [s for s in cam_skels if np.all(s[0] != -1)]

		# Check for skeletons
		# if len(cam_skels) == 0:
			# continue

		# Apply mask to image
		mask = cam.get_person() > 0
		# if mask is False:
		if 1:
			if len(cam_skels) > 0:
				# cam.colorIm = display_skeletons(cam.colorIm[:,:,2], skel_msr_im, (255,), skel_type='Kinect')[:,:,None]
				cam.colorIm[:,:,1] = display_skeletons(cam.colorIm[:,:,2], skel2depth(cam_skels[0], cam.depthIm.shape), (255,), skel_type='Kinect')

			## Max P=31 for LBPs becuase of datatype
			# tmp = local_binary_pattern(-cam.depthIm, 1, 10)#*(cam.foregroundMask>0)
			# embed()
			# tmp = local_occupancy_pattern(cam.depthIm, 31, 20, px_diff_thresh=100)*(cam.foregroundMask>0)

			# cv2.imshow("LBP", np.abs(tmp.astype(np.float))/float(tmp.max()))
			cam.colorIm = cam.colorIm[:,:,[0,2,1]]
			cam.visualize()
			continue

		# Anonomize
		# c_masked = cam.colorIm*mask[:,:,None]
		# d_masked = cam.depthIm*mask
		# c_masked_neg = cam.colorIm*(-mask[:,:,None])

		im_depth =  cam.depthIm
		if get_color:
			im_color = cam.colorIm
			im_color *= mask[:,:,None]
			im_color = np.ascontiguousarray(im_color)
			im_color = im_color[:,:,[2,1,0]]
		if len(cam_skels) > 0:
			skel_msr_xyz = cam_skels[0]
			skel_msr_im = skel2depth(cam_skels[0], cam.depthIm.shape)

		box = nd.find_objects(mask)[0]
		d = 20
		box = (slice(np.maximum(box[0].start-d, 0), \
				np.minimum(box[0].stop+d, height-1)), \
			   slice(np.maximum(box[1].start-d, 0), \
				np.minimum(box[1].stop+d, width-1)))

		# Face and skin detection
		if get_color:
			face_detector.run(im_color[box])
			im_skin = rgb2lab(cam.colorIm[box].astype(np.int16))[:,:,1]
			# im_skin = skimage.exposure.equalize_hist(im_skin)
			# im_skin = skimage.exposure.rescale_intensity(im_skin, out_range=[0,1])
			im_skin *= im_skin > face_detector.min_threshold
			im_skin *= im_skin < face_detector.max_threshold
			# im_skin *= face_detector>.068

			skin_match_c = nd.correlate(im_skin, hand_template)

			# Display Predictions - Color Based matching
			optima = peak_local_max(skin_match_c, min_distance=20, num_peaks=3, exclude_border=False)
			# Visualize
			if len(optima) > 0:
				optima_values = skin_match_c[optima[:,0], optima[:,1]]
				optima_thresh = np.max(optima_values) / 2
				optima = optima.tolist()

				for i,o in enumerate(optima):
					if optima_values[i] < optima_thresh:
						optima.pop(i)
						break
					joint = np.array(o) + [box[0].start, box[1].start]
					circ = np.array(circle(joint[0],joint[1], 5)).T
					circ = circ.clip([0,0], [height-1, width-1])
					cam.colorIm[circ[:,0], circ[:,1]] = (0,120 - 30*i,0)#(255*(i==0),255*(i==1),255*(i==2))
			markers = optima



		# ---------------- Tracking Algorithm ----------------
		# ---- Preprocessing ----
		if get_color:
			im_pos = rgbIm2PosIm(cam.depthIm*mask)[box] * mask[box][:,:,None]
		else:
			im_pos = cam.posIm[box]
		cam.depthIm[cam.depthIm==4500] = 0
		im_pos_mean = np.array([
							im_pos[:,:,0][im_pos[:,:,2]!=0].mean(),
							im_pos[:,:,1][im_pos[:,:,2]!=0].mean(),
							im_pos[:,:,2][im_pos[:,:,2]!=0].mean()
							], dtype=np.int16)

		# Zero-center
		if skel_current[0,2] == 0:
			skel_current += im_pos_mean
			skel_previous += im_pos_mean

		# Calculate Geodesic Extrema
		extrema = geodesic_extrema_MPI(im_pos, iterations=15, visualize=False)
		if len(extrema) > 0:
			for i,o in enumerate(extrema):
				joint = np.array(o) + [box[0].start, box[1].start]
				circ = np.array(circle(joint[0],joint[1], 5)).T
				circ = circ.clip([0,0], [height-1, width-1])
				cam.colorIm[circ[:,0], circ[:,1]] = (0,0,200-10*i)

		# Calculate Z-surface
		surface_map = nd.distance_transform_edt(-mask[box], return_distances=False, return_indices=True)

		# Only sample some of the points
		if 1:
			mask_interval = 1
			feature_radius = 10
		else:
			mask_interval = 3
			feature_radius = 2

		# Modify the box wrt the sampling
		box = (slice(box[0].start, box[0].stop, mask_interval),slice(box[1].start, box[1].stop, mask_interval))
		im_pos_full = im_pos.copy()
		im_pos = im_pos[::mask_interval,::mask_interval]
		box_height, box_width,_ = im_pos.shape

		skel_img_box = world2rgb(skel_current, cam.depthIm.shape) - [box[0].start, box[1].start, 0]
		skel_img_box = skel_img_box.clip([0,0,0], [im_pos.shape[0]-1, im_pos.shape[1]-1, 9999])

		feature_width = feature_radius*2+1
		all_features = [face_detector.face_position, optima, extrema]
		total_feature_count = np.sum([len(f) for f in all_features])

		# Loop through the rest of the constraints
		for _ in range(1):

			# ---- (Step 1A) Find feature coordespondences ----
			color_feature_displacement = feature_joint_displacements(skel_current, im_pos, all_features[1], features_joints[1], distance_thresh=500)
			depth_feature_displacement = feature_joint_displacements(skel_current, im_pos, all_features[2], features_joints[2], distance_thresh=500)

			# Alternative method: use kdtree
			## Calc euclidian distance between each pixel and all joints
			px_corr = np.zeros([im_pos.shape[0], im_pos.shape[1], len(skel_current)])
			for i,s in enumerate(skel_current):
				px_corr[:,:,i] = np.sqrt(np.sum((im_pos - s)**2, -1))# / joint_size[i]**2

			## Handle occlusions by argmax'ing over set of skel parts
			# visible_configurations = list(it.product([0,1], repeat=5))[1:]
			visible_configurations = [
										#[0,1,1,1,1],
										#[1,0,0,0,0],
										[1,1,1,1,1]
									]
			px_visibility_label = np.zeros([im_pos.shape[0], im_pos.shape[1], len(visible_configurations)], dtype=np.uint8)
			visible_scores = np.ones(len(visible_configurations))*np.inf
			# Try each occlusion configuration set
			for i,v in enumerate(visible_configurations):
				visible_joints = list(it.chain.from_iterable(skel_parts[np.array(v)>0]))
				px_visibility_label[:,:,i] = np.argmin(px_corr[:,:,visible_joints], -1)#.reshape([im_pos.shape[0], im_pos.shape[1]])
				visible_scores[i] = np.min(px_corr[:,:,visible_joints], -1).sum()
			# Choose best occlusion configuration
			occlusion_index = np.argmin(visible_scores)
			occlusion_configuration = visible_configurations[occlusion_index]
			occlusion_set = list(it.chain.from_iterable(skel_parts[np.array(visible_configurations[occlusion_index])>0]))
			# Choose label for pixels based on occlusion configuration
			px_label = px_visibility_label[:,:,occlusion_index]*mask[box]
			px_label_flat = px_visibility_label[:,:,occlusion_index][mask[box]].flatten()

			visible_joints = [1 if x in occlusion_set else 0 for x in range(len(skel_current))]

			# Project distance to joint's radius
			px_joint_displacement = skel_current[px_label_flat] - im_pos[mask[box]]
			px_joint_magnitude = np.sqrt(np.sum(px_joint_displacement**2,-1))
			joint_mesh_pos = skel_current[px_label_flat] + px_joint_displacement*(joint_size[px_label_flat]/px_joint_magnitude)[:,None]
			px_joint_displacement = joint_mesh_pos - im_pos[mask[box]]

			 # Calc the correspondance change in position for each joint
			correspondence_displacement = np.zeros([len(skel_current), 3])
			ii = 0
			for i,_ in enumerate(skel_current):
				if i in occlusion_set:
					labels = px_label_flat==i
					correspondence_displacement[i] = np.mean(px_joint_displacement[px_label_flat==ii], 0)
					# correspondence_displacement[ii] = np.sum(px_joint_displacement[px_label_flat==ii], 0)
					ii+=1
			correspondence_displacement = np.nan_to_num(correspondence_displacement)

			# ---- (Step 2) Update pose state, x ----
			lambda_p = .0
			lambda_c = .3
			lambda_cf = .3
			lambda_df = .0
			skel_prev_difference = (skel_current - skel_previous)
			# embed()
			skel_current = skel_previous \
							+ lambda_p  * skel_prev_difference \
							- lambda_c  * correspondence_displacement\
							- lambda_cf * color_feature_displacement\
							- lambda_df * depth_feature_displacement

			# ---- (Step 3) Add constraints ----
			# A: Link lengths / geometry
			skel_current = link_length_constraints(skel_current, constraint_links, constraint_values, alpha=.5)
			skel_current = geometry_constraints(skel_current, joint_size, alpha=0.5)
			# skel_current = collision_constraints(skel_current, constraint_links)

			skel_img_box = (world2rgb(skel_current, cam.depthIm.shape) - [box[0].start, box[1].start, 0])/mask_interval
			skel_img_box = skel_img_box.clip([0,0,0], [cam.depthIm.shape[0]-1, cam.depthIm.shape[1]-1, 9999])
			# B: Ray-cast constraints
			skel_current, skel_img_box = ray_cast_constraints(skel_current, skel_img_box, im_pos_full, surface_map, joint_size)

		# # Map back from mask to image
		skel_img = skel_img_box + [box[0].start, box[1].start, 0]

		# Update for next round
		skel_previous = skel_current.copy()
		# skel_previous = skel_init.copy()

		# ---------------------Accuracy --------------------------------------

		# Compute accuracy wrt standard Kinect data
		# skel_im_error = skel_msr_im[:,[1,0]] - skel_img[[0,2,3,4,5,6,7,8,9,10,11,12,13,14],:2]
		skel_current_kinect = convert_to_kinect(skel_current)
		try:
			skel_msr_im_box = np.array([skel_msr_im[:,1]-box[0].start,skel_msr_im[:,0]-box[1].start]).T.clip([0,0],[box_height-1, box_width-1])
			skel_xyz_error = im_pos[skel_msr_im_box[:,0],skel_msr_im_box[:,1]] - skel_current_kinect#skel_current[[0,2,3,4,5,6,7,8,9,10,11,12],:]
			skel_l2 = np.sqrt(np.sum(skel_xyz_error**2, 1))
			# print skel_l2
			skel_correct = np.nonzero(skel_l2 < 150)[0]
			accuracy_measurements['per_joint'] += [skel_l2]
			accuracy_measurements['overall'] += [len(skel_correct)/float(len(skel_current_kinect))*100]
			print "{0:0.2f}% joints correct".format(len(skel_correct)/float(len(skel_current_kinect))*100)
			print "Overall accuracy: ", np.mean(accuracy_measurements['overall'])
		except:
			pass

		print "Visible:", visible_joints

		# ----------------------Visualization-------------------------------------
		# l = line(skel_img_box[joint][0], skel_img_box[joint][1], features[feat][0], features[feat][1])
		# skimage.draw.set_color(cam.colorIm[box], l, (255,255,255))

		# Add circles to image
		cam.colorIm = np.ascontiguousarray(cam.colorIm)
		if 0:#get_color:
			cam.colorIm = display_skeletons(cam.colorIm, skel_img[:,[1,0,2]]*np.array(visible_joints)[:,None], (0,255,), skel_type='Other', skel_contraints=constraint_links)
			for i,s in enumerate(skel_img):
				# if i not in skel_correct:
				if i not in occlusion_set:
					c = circle(s[0], s[1], 5)
					cam.colorIm[c[0], c[1]] = (255,0,0)
			# cam.colorIm = display_skeletons(cam.colorIm, world2rgb(skel_init+im_pos_mean, [240,320])[:,[1,0]], skel_type='Other', skel_contraints=constraint_links)

		if 1:
			if len(face_detector.face_position) > 0:
				for (x, y) in face_detector.face_position:
					pt1 = (int(y)+box[1].start-15, int(x)+box[0].start-15)
					pt2 = (pt1[0]+int(15), pt1[1]+int(15))
					cv2.rectangle(cam.colorIm, pt1, pt2, (255, 0, 0), 3, 8, 0)
			if len(cam_skels) > 0:
				# cam.colorIm = display_skeletons(cam.colorIm[:,:,2], skel_msr_im, (255,), skel_type='Kinect')[:,:,None]
				cam.colorIm[:,:,1] = display_skeletons(cam.colorIm[:,:,2], skel_msr_im, (255,), skel_type='Kinect')
			cam.visualize()
			cam.depthIm = local_binary_pattern(cam.depthIm*cam.foregroundMask, 50, 10)
			cv2.imshow("Depth", cam.depthIm/float(cam.depthIm.max()))
			# cam2.visualize()
			# embed()

		# 3D Visualization
		if 0:
			from mayavi import mlab
			# figure = mlab.figure(1, bgcolor=(0,0,0), fgcolor=(1,1,1))
			figure = mlab.figure(1, bgcolor=(1,1,1))
			figure.scene.disable_render = True
			mlab.clf()
			mlab.view(azimuth=-45, elevation=45, roll=0, figure=figure)
			mlab.points3d(-skel_current[:,1], -skel_current[:,0], skel_current[:,2], scale_factor=100., color=(.5,.5,.5))
			for c in constraint_links:
				x = np.array([skel_current[c[0]][0], skel_current[c[1]][0]])
				y = np.array([skel_current[c[0]][1], skel_current[c[1]][1]])
				z = np.array([skel_current[c[0]][2], skel_current[c[1]][2]])
				mlab.plot3d(-y,-x,z, tube_radius=25., color=(1,0,0))
			figure.scene.disable_render = False

		# 3-panel view
		if 0:
			subplot(2,2,1)
			scatter(skel_current[:,1], skel_current[:,2]);
			for i,c in enumerate(constraint_links):
				plot([skel_current[c[0],1], skel_current[c[1],1]],[skel_current[c[0],2], skel_current[c[1],2]])
			axis('equal')

			subplot(2,2,3)
			scatter(skel_current[:,1], -skel_current[:,0]);
			for i,c in enumerate(constraint_links):
				plot([skel_current[c[0],1], skel_current[c[1],1]],[-skel_current[c[0],0], -skel_current[c[1],0]])
			axis('equal')

			subplot(2,2,4)
			scatter(skel_current[:,2], -skel_current[:,0]);
			for i,c in enumerate(constraint_links):
				plot([skel_current[c[0],2], skel_current[c[1],2]],[-skel_current[c[0],0], -skel_current[c[1],0]])
			axis('equal')
			# show()

		# ------------------------------------------------------------

		video_writer.write(cam.colorIm[:,:,[2,1,0]])
		frame_count+=1
	print 'Done'

if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	parser.add_option('-l', '--learn', dest='learn', action="store_true", default=False, help='Training phase')
	(opt, args) = parser.parse_args()

	main(visualize=opt.viz, learn=opt.learn)



# Tracking extra
# if 0:
	# Based on geodesic distance
	# skel_img = world2rgb(skel_current, cam.depthIm.shape) - [box[0].start, box[1].start, 0]
	# skel_img = skel_img.clip([0,0,0], [mask[box].shape[0]-1, mask[box].shape[1]-1, 999])
	# for i,s in enumerate(skel_img):
	# 	_, geodesic_map = geodesic_extrema_MPI(cam.depthIm[box]*mask[box], centroid=[s[0],s[1]], iterations=1, visualize=True)
	# 	px_corr[:,:,i] = geodesic_map + (-mask[box])*9999


