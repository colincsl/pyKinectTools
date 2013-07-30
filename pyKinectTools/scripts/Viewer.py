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
import skimage.draw as draw
import cv, cv2
from pylab import *

from pyKinectTools.dataset_readers.KinectPlayer import KinectPlayer, display_help

""" Debugging """
from IPython import embed


if 0:
	## ICU anonomization test
	im = cam.colorIm.copy()
	# im[::2,::2,0] = 255-im[::2,::2,0]
	# im[::2,::2,1] = 128-im[::2,::2,1]
	# im[::2,::2,2] = im[::2,::2,2]-64
	tmp = np.random.randint(0, 255, cam.colorIm.shape).astype(np.uint8)
	im_in = cam.colorIm[:,:,[2,1,0]]
	im_tmp = im_in+tmp
	im_out = im_tmp-tmp

	subplot(1,4,1)
	imshow(im_in)
	subplot(1,4,2)
	subplot(1,4,1)
	title('Original')
	subplot(1,4,2)
	title('Template')
	imshow(tmp)
	subplot(1,4,3)
	imshow(im_tmp)
	title('Result (Anonomized)')
	subplot(1,4,4)
	title('De-anonomized')
	imshow(im_out)
	show()


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
		cam.set_bg_model(bg_type='box', param=2600)

	# Test HOG pedestrian detector
	# cv2.HOGDescriptor_getDefaultPeopleDetector()
	# imshow(color[people])

	framerate = 1
	while cam.next(framerate):

		if 0:
			people_all = list(cv.HOGDetectMultiScale(cv.fromarray(cam.colorIm.mean(-1).astype(np.uint8)), cv.CreateMemStorage(0), hit_threshold=-1.5))
			print len(people_all), " people detected"
			for i,people in enumerate(people_all):
				# embed()
				try:
					print people
					# tmp = cam.colorIm[people[0][0]:people[0][0]+people[1][0], people[0][1]:people[0][1]+people[1][1]]
					# tmp = cam.colorIm[people[0][1]:people[0][1]+people[1][1], people[0][0]:people[0][0]+people[1][0]]
					cam.colorIm[people[0][1]:people[0][1]+people[1][1], people[0][0]:people[0][0]+people[1][0]] += 50
					# cv2.imshow("Body2 "+str(i), tmp)
					# subplot(4, len(people_all)/4, i + 1)
					# imshow(tmp)

					# tmp = cam.colorIm[people[0][0]:people[0][0]+people[0][1], people[1][0]:people[1][0]+people[1][1]]
					# cv2.imshow("Body1 "+str(i), tmp)
					# tmp = cam.colorIm[people[1][0]:people[1][0]+people[1][1], people[0][0]:people[0][0]+people[0][1]]
					# print tmp
					# cv2.imshow("Body "+str(i), tmp)
				except:
					pass
			# show()

		if anonomization and cam.mask is not None:

			mask = (sm.imresize(cam.mask, [480,640]) > 0).copy()
			mask[40:170, 80:140] = True # Suchi's body
			px = draw.circle(145,285, 40)
			mask[px] = True

			# Version 1 - Blur + Circle
			color = cam.colorIm.copy()
			color[mask] = cv2.GaussianBlur(color, (29,29), 50)[mask]

			subplot(1,3,1)
			imshow(color)

			# Version 2 - Noise on mask
			color = cam.colorIm.copy()
			noise_key = np.random.randint(0, 255, cam.colorIm.shape).astype(np.uint8)
			color[mask] = color[mask]+noise_key[mask]

			subplot(1,3,2)
			imshow(color)

			# Version 3 - All noise
			color = cam.colorIm.copy()
			color = color+noise_key

			subplot(1,3,3)
			imshow(color)

			show()



			# cam.colorIm *= mask[:,:,None]
		cam.visualize(color=True, depth=True, text=True, colorize=True, depth_bounds=[0,5000])
		# cam.visualize(color=True, depth=True, text=False, colorize=False, depth_bounds=None)

	print 'Done'

	# Pause at the end
	embed()


if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-a', '--anon', dest='anon', action="store_true", default=False, help='Enable anonomization')
	(opt, args) = parser.parse_args()

	main(opt.anon)

