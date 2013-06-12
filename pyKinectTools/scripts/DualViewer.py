"""
Main file for training multi-camera pose
"""

import os
import optparse
import time
import cPickle as pickle
import numpy as np
from skimage import color

# from rgbdActionDatasets.dataset_readers.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.dataset_readers.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.utils.DepthUtils import world2depth, depthIm2XYZ, skel2depth, depth2world
from pyKinectTools.utils.SkeletonUtils import display_skeletons, transform_skels, kinect_to_msr_skel

""" Debugging """
from IPython import embed
np.seterr(all='ignore')

# -------------------------MAIN------------------------------------------

def main(visualize=True):
	n_cameras = 1
	cam = KinectPlayer(base_dir='./', device=1, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=False, fill_images=False)
	if n_cameras == 2:
		cam2 = KinectPlayer(base_dir='./', device=2, bg_subtraction=True, get_depth=True, get_color=get_color, get_skeleton=get_skel, fill_images=fill)
	# Transformation matrix from first to second camera
	try:
		data = pickle.load(open("Registration.dat", 'r'))
		transform_c1_to_c2 = data['transform']
		transform = True
	except:
		transform = False
		pass

	current_frame = 0
	all_joint_ims_z = []
	all_joint_ims_c = []
	framerate = 1
	while cam.next():
		# print "Frame ", current_frame
		# Update frames
		if n_cameras == 2:
			cam2.next()
			# cam2.sync_cameras(cam)
		current_frame+=1
		if current_frame%framerate != 0:
			# current_frame += 1
			continue

		# Transform skels from cam1 to cam2
		# cam_skels = [np.array(cam.users[s]['jointPositions'].values()) for s in cam.users.keys()]
		# cam_skels = [np.array(cam.users[s]['jointPositions'].values()) for s in cam.users]
		# Get rid of bad skels
		# cam_skels = [s for s in cam_skels if np.all(s[0] != -1)]

		# if len(cam_skels) == 0:
		# 	continue


		# Save images
		if 0:
			joint_ims_z = []
			joint_ims_c = []
			dx = 10
			skel_tmp = skel2depth(cam_skels[0], [240,320])
			for j_pos in skel_tmp:
				# embed()
				joint_ims_z += [cam.depthIm[j_pos[0]-dx:j_pos[0]+dx, j_pos[1]-dx:j_pos[1]+dx]]
				joint_ims_c += [cam.colorIm[j_pos[0]-dx:j_pos[0]+dx, j_pos[1]-dx:j_pos[1]+dx]]
			if len(joint_ims_z) > 0:
				all_joint_ims_z += [joint_ims_z]
				all_joint_ims_c += [joint_ims_c]

		if 1:
			# if transform:
				# cam2_skels = transform_skels(cam_skels, transform_c1_to_c2, 'image')

			# try:
				# depth = cam2.get_person()
				# if learn:
				# 	rf.add_frame(depth, cam2_skels[0])
				# else:
				# 	rf.infer_pose(depth)
			# except:
				# pass
			if visualize:
				# cam2.depthIm = display_skeletons(cam2.depthIm, cam2_skels[0], (5000,), skel_type='Low')
				# skel1 = kinect_to_msr_skel(skel2depth(cam_skels[0], [240,320]))
				# cam.depthIm = display_skeletons(cam.depthIm, skel1, (5000,), skel_type='Low')
				# embed()
				cam.visualize(color=True, depth=True, text=True, colorize=True, depth_bounds=[500,3500])
				if n_cameras == 2:
					cam2.visualize(color=True, depth=True)


	embed()


	print 'Done'

if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=True, help='Enable visualization')
	(opt, args) = parser.parse_args()

	main(opt.viz)


	# '''Profiling'''
	# cProfile.runctx('main()', globals(), locals(), filename="ShowSkeletons.profile")
