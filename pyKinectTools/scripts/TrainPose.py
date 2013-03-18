"""
Main file for training multi-camera pose
"""

import os
import optparse
import time
import cPickle as pickle
import numpy as np
from skimage import color

from pyKinectTools.utils.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.utils.DepthUtils import world2depth, depthIm2XYZ, skel2depth, depth2world
from pyKinectTools.utils.SkeletonUtils import display_skeletons, transform_skels, kinect_to_msr_skel
# from pyKinectTools.algs.GeodesicSkeleton import generateKeypoints, distance_map
from pyKinectTools.algs.RandomForestPose import RFPose

""" Debugging """
from IPython import embed
np.seterr(all='ignore')

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

	# Get random forest parameters
	if learn:
		rf = RFPose(offset_max=100, feature_count=300)
	else:
		rf = RFPose()
		rf.load_forest()

	ii = 0
	# cam.next(200)
	all_joint_ims_z = []
	all_joint_ims_c = []
	while cam.next() and ii < 200:
		# Update frames
		cam2.sync_cameras(cam)
		if ii%10 != 0:
			ii += 1
			continue

		# Transform skels from cam1 to cam2
		cam_skels = [np.array(cam.users[s]['jointPositions'].values()) for s in cam.users.keys()]
		# Get rid of bad skels
		cam_skels = [s for s in cam_skels if np.all(s[0] != -1)]

		if len(cam_skels) == 0: 
			continue
		ii+=1	

		# Save images
		if 1:
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

		if 0:
			cam2_skels = transform_skels(cam_skels, transform_c1_to_c2, 'image')

			try:
				depth = cam2.get_person()
				if learn:
					rf.add_frame(depth, cam2_skels[0])
				else:
					rf.infer_pose(depth)

				if visualize:
					cam2.depthIm = display_skeletons(cam2.depthIm, cam2_skels[0], (5000,), skel_type='Low')
					skel1 = kinect_to_msr_skel(skel2depth(cam_skels[0], [240,320]))
					cam.depthIm = display_skeletons(cam.depthIm, skel1, (5000,), skel_type='Low')
					cam.visualize()
					cam2.visualize()
			except:
				pass


	embed()
	if learn:
		print "Starting forest"
		rf.learn_forest()

	print 'Done'

if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	parser.add_option('-l', '--learn', dest='learn', action="store_true", default=False, help='Training phase')
	parser.add_option('-i', '--infer', dest='infer', action="store_true", default=False, help='Training phase')
	(opt, args) = parser.parse_args()

	if len(args) > 0:
		print "Wrong input argument"
	elif opt.infer==False and opt.learn==False:
		print "You must supply the program with either -l (learn parameters) or -i (infer pose)."
	else:
		if opt.infer:
			main(visualize=opt.viz, learn=False)
		else:
			main(visualize=opt.viz, learn=True)

	# '''Profiling'''
	# cProfile.runctx('main()', globals(), locals(), filename="ShowSkeletons.profile")
