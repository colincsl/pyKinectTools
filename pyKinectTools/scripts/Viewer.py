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

# -------------------------MAIN------------------------------------------

def main(visualize=True):
	n_cameras = 1
	cam = KinectPlayer(base_dir='./', device=1, bg_subtraction=False, get_depth=True, get_color=True, get_skeleton=False, fill_images=False)

	framerate = 1
	while cam.next(framerate):
		cam.visualize(color=True, depth=True, text=True, colorize=True, depth_bounds=[500,3500])

	embed()

	print 'Done'

if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=True, help='Enable visualization')
	(opt, args) = parser.parse_args()

	main(opt.viz)