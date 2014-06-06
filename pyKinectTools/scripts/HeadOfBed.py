"""
Main file for training multi-camera pose
"""

import os
import optparse
import time
import cPickle as pickle
import numpy as np
from skimage import color
import cv2
import pcl
import itertools as it

# import rospy
# from sensor_msgs.msg import PointCloud2
# from std_msgs.msg import String

# pub_str = rospy.Publisher('str', String)
# pub_cloud = rospy.Publisher('cloud', PointCloud2)
# rospy.init_node('pyPCL')


# from rgbdActionDatasets.dataset_readers.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.dataset_readers.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.utils.DepthUtils import world2depth, depthIm2XYZ, skel2depth, depth2world
from pyKinectTools.utils.SkeletonUtils import display_skeletons, transform_skels, kinect_to_msr_skel
from pyKinectTools.algs.BackgroundSubtraction import extract_people

""" Debugging """
from IPython import embed

num_to_rgb = np.array([
	[1,0,0],
	[0,1,0],
	[0,0,1],
	[1,1,0],
	[0,1,1],
	[1,0,1],
	[1,1,1]])

# -------------------------MAIN------------------------------------------

def find_plane(cloud, dist_thresh=10, max_iter=500):
	seg = cloud.make_segmenter_normals(ksearch=10)
	seg.set_optimize_coefficients (True);
	seg.set_model_type (pcl.SACMODEL_NORMAL_PLANE)
	seg.set_distance_threshold(dist_thresh)
	seg.set_method_type (pcl.SAC_RANSAC)
	seg.set_max_iterations (max_iter)
	indices, model = seg.segment()
	# idx = np.unravel_index(indices,cam.depthIm.shape)

	return indices,model


def main(visualize=True):
	n_cameras = 1
	cam = KinectPlayer(base_dir='./', device=1, bg_subtraction=True, get_depth=True,
	get_color=True, get_skeleton=False, background_model='box', background_param=3200)
	
	# embed()
	cloud = pcl.PointCloud()
	angle_measurements = []
	dist_thresh = 10
	max_iter = 500
	n_planes = 3

	framerate = 1
	cam.play_speed = 300
	while cam.next(framerate):
		# mask = cam.get_person() > 0
		# embed()
		id_im, id_slices, ids, id_px = extract_people(cam.depthIm, gradThresh=50)
		# id_max = np.argmin([np.abs(cam.depthIm[0]/2 - (x[1].stop+x[1].start)/2.) for x in id_slices])
		id_max = np.argmin([np.abs(320/2 - (x[1].stop+x[1].start)/2.) for x in id_slices])
		mask = id_im==(id_max+1)
		
		cam.depthIm *= mask

		planes_idx = []
		planes_models = []
		plane_im = np.repeat(cam.depthIm[:,:,None], 3, -1).reshape([-1,3]).copy()
		# plane_im_out = np.repeat(cam.depthIm[:,:,None]*0, 3, -1).reshape([-1,3]).copy()
		plane_im_out = np.zeros([cam.depthIm.shape[0]*cam.depthIm.shape[1], 3], np.float)

		# Find planes
		for i in range(n_planes):
			# Get valid points
			pts = cam.camera_model.im2PosIm(plane_im[:,0].reshape(cam.depthIm.shape)).reshape([-1,3])
			nonzero_idx = np.nonzero((pts[:,2]>0)*(pts[:,2]!=1000) )
			cloud.from_array(pts[nonzero_idx].astype(np.float32))

			# Find plane
			indices, model = find_plane(cloud, dist_thresh, max_iter)
			print "Plane:", model, len(indices)
			# if len(indices) < 5000:
				# continue
			planes_idx += [indices]
			planes_models += [model]
			
			# Remove labeled points and highlight on image
			tmp = plane_im[nonzero_idx]
			tmp[(np.array(indices),)] = num_to_rgb[i]*1000
			plane_im[nonzero_idx] = tmp
			# plane_im_out[nonzero_idx[(np.array(indices),)]] = tmp
			plane_im_out[nonzero_idx[0][indices]] = num_to_rgb[i]*1000
			


		angle_measurements = []
		for combos in it.combinations(planes_models, 2):
			angle_measurements += [np.arccos(np.dot(combos[0][:3], combos[1][:3]))*180./np.pi]
		print "Current angle:", angle_measurements
		# angle_measurements += [np.arccos(np.dot(planes_models[0][:3], planes_models[1][:3]))*180./np.pi]
		# print "Current angle:", angle_measurements[-1]
		# print "Average angle:", np.mean(angle_measurements)
		print ""
		
		# tmp = plane_im[nonzero_idx]
		# tmp[(np.array(planes_idx[1]),)] = np.array([0, 2000, 0])
		# plane_im[nonzero_idx] = tmp

		plane_im = plane_im.reshape([240,320,3])
		plane_im_out = plane_im_out.reshape([240,320,3])
		# cv2.imshow("dd", plane_im/float(plane_im.max()))
		cv2.imshow("dd", (plane_im_out/float(plane_im_out.max()))*100. + cam.colorIm/300.)
		# cv2.waitKey(50)

		# cam.visualize(color=True, depth=True, text=True, colorize=True, depth_bounds=[500,3500])
		cam.visualize(color=False, depth=True, text=True, colorize=False, depth_bounds=[500,3500])
		# embed()

	embed()

	print 'Done'

if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=True, help='Enable visualization')
	(opt, args) = parser.parse_args()

	main(opt.viz)