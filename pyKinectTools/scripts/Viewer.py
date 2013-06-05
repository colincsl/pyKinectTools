"""
Main file for viewing data
"""

import os
import optparse
import time
import cPickle as pickle
import numpy as np
from skimage import color
import scipy.misc as sm

from pyKinectTools.dataset_readers.KinectPlayer import KinectPlayer, display_help
# from pyKinectTools.utils.SkeletonUtils import display_skeletons, transform_skels, kinect_to_msr_skel

""" Debugging """
from IPython import embed

# -------------------------MAIN------------------------------------------

def main(anonomization=False):
	# Setup kinect data player
	cam = KinectPlayer(base_dir='./', bg_subtraction=False, get_depth=True, get_color=True, get_skeleton=False)

	if anonomization:
		''' bg_type can be:
				'box'[param=max_depth]
				'static'[param=background]
				'mean'
				'median'
				'adaptive_mog'

				See BasePlayer for more details
		'''
		cam.set_bg_model(bg_type='box', param=2500)


	framerate = 1
	while cam.next(framerate):

		if anonomization and  cam.mask is not None:
			mask = sm.imresize(cam.mask, [480,640]) == 0
			cam.colorIm *= mask[:,:,None]
		cam.visualize(color=True, depth=True, text=True, colorize=True, depth_bounds=[0,5000])
		# cam.visualize(color=True, depth=True, text=False, colorize=False, depth_bounds=None)

	print 'Done'

	# Pause at the end
	embed()


if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-a', '--anon', dest='anon', action="store_true", default=True, help='Enable anonomization')
	(opt, args) = parser.parse_args()

	main(opt.anon)

