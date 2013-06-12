"""
Main file for viewing data
"""

import os
import time
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

try:
	import serial
	ser = serial.Serial(port='/dev/tty.usbmodemfa131', baudrate=9600)
	# Turn on light
	ser.write('3')
except:
	raise Exception, "Please install PySerial (pip install pyserial)"

# -------------------------MAIN------------------------------------------

def main(anonomization=False):
	# Setup kinect data player
	cam = KinectPlayer(base_dir='./', bg_subtraction=False, get_depth=True, get_color=True, get_skeleton=False)
	recording_enabled = True
	button_pressed = False

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

		while ser.inWaiting() > 0:
			button_current = ser.readline()
			print button_current, button_pressed
			if button_pressed != button_current and button_current:
				recording_enabled = False
				recording_time = time.time()
				# Turn off light
				ser.write('4')
				print 'Off'
			button_pressed = button_current

		if not recording_enabled:
			if time.time() - recording_time > 5:
				recording_enabled = True
				ser.write('3')
				print 'On'
			else:
				continue

		if anonomization and  cam.mask is not None:
			mask = cam.mask == 0
			if cam.colorIm.shape[:2] != cam.mask.shape:
				mask = sm.imresize(mask, [480,640])
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

