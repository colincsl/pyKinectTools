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

# from rgbdActionDatasets.dataset_readers.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.dataset_readers.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.utils.DepthUtils import world2depth, depthIm2XYZ, skel2depth, depth2world
from pyKinectTools.utils.SkeletonUtils import display_skeletons, transform_skels, kinect_to_msr_skel

""" Debugging """
from IPython import embed

# -------------------------MAIN------------------------------------------

def main(visualize=True):
	n_cameras = 1
	cam = KinectPlayer(base_dir='./', device=1, bg_subtraction=True, get_depth=True,
	get_color=True, get_skeleton=False, background_model='box', background_param=2500)
	
	cloud = pcl.PointCloud()
	angle_measurements = []
	
	framerate = 1
	while cam.next(framerate):
		# cam.depthIm[cam.depthIm>2500] = 0
		# embed()
		mask = cam.get_person() > 0
		cam.depthIm *= mask

		planes_idx = []
		planes_models = []
		plane_im = np.repeat(cam.depthIm[:,:,None], 3, -1).reshape([-1,3]).copy()

		# Form pointcloud (top)
		pts = cam.camera_model.im2PosIm(cam.depthIm).reshape([-1,3])
		pts[:pts.shape[0]*2/3] *= 0
		nonzero_idx = np.nonzero(pts[:,2]>0)
		cloud.from_array(pts[nonzero_idx].astype(np.float32))

		# Find plane
		seg = cloud.make_segmenter_normals(ksearch=20)
		seg.set_optimize_coefficients (True);
		seg.set_model_type (pcl.SACMODEL_NORMAL_PLANE)
		seg.set_distance_threshold(50)
		seg.set_method_type (pcl.SAC_RANSAC)
		seg.set_max_iterations (500)
		indices, model = seg.segment()
		idx = np.unravel_index(indices,cam.depthIm.shape)

		planes_idx += [indices]
		planes_models += [model]
		print "Plane 1:", model

		tmp = plane_im[nonzero_idx]
		tmp[(np.array(planes_idx[0]),)] = np.array([1000, 0, 0])
		plane_im[nonzero_idx] = tmp


		# Form pointcloud (bottom)
		pts = cam.camera_model.im2PosIm(cam.depthIm).reshape([-1,3])
		pts[pts.shape[0]*2/3::] *= 0
		nonzero_idx = np.nonzero(pts[:,2]>0)
		cloud.from_array(pts[nonzero_idx].astype(np.float32))

		# Find plane
		seg = cloud.make_segmenter_normals(ksearch=20)
		seg.set_optimize_coefficients (True);
		seg.set_model_type (pcl.SACMODEL_NORMAL_PLANE)
		seg.set_distance_threshold(50)
		seg.set_method_type (pcl.SAC_RANSAC)
		seg.set_max_iterations (500)
		indices, model = seg.segment()
		idx = np.unravel_index(indices,cam.depthIm.shape)

		planes_idx += [indices]
		planes_models += [model]
		print "Plane 2:", model

		angle_measurements += [np.arccos(np.dot(planes_models[0][:3], planes_models[1][:3]))*180./np.pi]
		print "Current angle:", angle_measurements[-1]
		print "Average angle:", np.mean(angle_measurements)
		print ""
		# embed()
		
		tmp = plane_im[nonzero_idx]
		tmp[(np.array(planes_idx[1]),)] = np.array([0, 2000, 0])
		plane_im[nonzero_idx] = tmp

		plane_im = plane_im.reshape([240,320,3])
		cv2.imshow("dd", plane_im/float(plane_im.max()))
		cv2.waitKey(50)



		cam.visualize(color=True, depth=True, text=True, colorize=True, depth_bounds=[500,3500])
	embed()

	print 'Done'

if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=True, help='Enable visualization')
	(opt, args) = parser.parse_args()

	main(opt.viz)