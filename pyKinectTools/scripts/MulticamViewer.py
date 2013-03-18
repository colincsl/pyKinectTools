'''
Main file for displaying depth/color/skeleton information and extracting features
'''


import os
import optparse
import time
import cPickle as pickle
import numpy as np
import scipy.ndimage as nd
from skimage import color
from skimage.segmentation import felzenszwalb
import cv2
from pyKinectTools.utils.KinectPlayer import KinectPlayer, display_help
from pyKinectTools.utils.DepthUtils import *#world2depth, depthIm2XYZ, skel2depth, depth2world
# from pyKinectTools.algs.BackgroundSubtraction import AdaptiveMixtureOfGaussians, fillImage, extract_people
from pyKinectTools.algs.BackgroundSubtraction import extract_people
from pyKinectTools.utils.SkeletonUtils import display_skeletons
from pyKinectTools.algs.GeodesicSkeleton import *


''' Debugging '''
from IPython import embed
np.seterr(all='ignore')

# -------------------------MAIN------------------------------------------

def main(get_depth, get_color, get_skeleton, visualize):
	# fill = True
	fill = False
	get_color = True
	# VP = KinectPlayer(base_dir='./', device=device, get_depth=get_depth, get_color=get_color, get_skeleton=get_skeleton, get_mask=get_mask)
	# VP = KinectPlayer(base_dir='./', device=1, bg_subtraction=True, get_depth=get_depth, get_color=get_color, get_skeleton=get_skeleton, fill_images=fill)	
	VP = KinectPlayer(base_dir='./', device=2, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=False, fill_images=fill)	
	cam_count = 1
	if cam_count == 2:
		VP2 = KinectPlayer(base_dir='./', device=2, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=False, fill_images=fill)	
		# Transformation matrix from first to second camera
		# data = pickle.load(open("./Registration.dat", 'r'))
		# transform_c1_to_c2 = data['transform']

	# print 'aaaaaaaaaaaaaaaa'
	# embed()

	while VP.next():
		# VP.update_background()
		# Transform skels from cam1 to cam2
		if get_skeleton:
			VP_skels = [np.array(VP.users[s]['jointPositions'].values()) for s in VP.users.keys()]
		if cam_count == 2:
			VP2.sync_cameras(VP)
			if get_skeleton:
				VP2_skels = transform_skels(VP_skels, transform_c1_to_c2)

		VP.colorIm = VP.colorIm[:,:,[2,1,0]]

		# im_tmp = np.dstack([VP.depthIm/float(VP.depthIm.max()),\
		# 					VP.depthIm/float(VP.depthIm.max()),\
		# 					VP.colorIm.mean(-1)/255])
		# im = felzenszwalb(im_tmp, scale=255)
		# im = felzenszwalb(VP.depthIm/float(VP.depthIm.max()), scale=255)

		# im = VP.depthIm/float(VP.depthIm.max()) * VP.
		# im = felzenszwalb(im, scale=255)
		# cv2.imshow("seg", im/float(im.max()))

		# cv2.waitKey(10)
		# embed()

		# Geodesic extrema
		if 0:
			if VP.foregroundMask is not None:
				im = VP.depthIm	* (VP.foregroundMask>0)
				# cv2.imshow("im--", im/float(im.max()))
				if 1:#(im>0).sum() > 1000:
					# Extract person
					labelMask, boxes, labels, px_counts= extract_people(im, VP.foregroundMask, minPersonPixThresh=3000)
					# print labels, px_counts
					if len(labels) > 0:
						max_ind = np.argmax(px_counts)
						mask = labelMask==max_ind+1
						im[-mask] = 0

						edge_thresh = 200#10
						gradients = np.gradient(im)
						mag = np.sqrt(gradients[0]**2+gradients[1]**2)
						im[mag>edge_thresh] = 0

						# Segmentation experiment
						# im_s = VP.depthIm/float(VP.depthIm.max())
						# im_s = felzenszwalb(im, scale=255)
						# cv2.imshow("seg", im_s/float(im_s.max()))

						x,y = nd.center_of_mass(im)
						if im[x,y] == 0:
							tmp = np.nonzero(im>0)
							x,y = [tmp[0][0], tmp[1][0]]
						# min_map = geodesic_extrema_MPI(im, centroid=[x,y], iterations=15)
						extrema = geodesic_extrema_MPI(im, centroid=[x,y], iterations=15)
						# embed()

						# for e, i in zip(extrema, range(20)):
						# 	if 0 < i < 6:
						# 		box_color = [255, 0, 0]
						# 	elif i==0:
						# 		box_color = [0, 0, 255]
						# 	else:
						# 		box_color = [255,255,255]
						# 	VP.colorIm[e[0]-4:e[0]+5, e[1]-4:e[1]+5] = box_color

						# cv2.imshow("ExtremaB", min_map/float(min_map.max()))
						# cv2.waitKey(500)
						cv2.waitKey(20)

		# cv2.imshow("bg model", VP.backgroundModel/float(VP.backgroundModel.max()))
		# cv2.imshow("foregroundMask", ((VP.foregroundMask>0)*255).astype(np.uint8))
		if visualize:
			if cam_count == 2:
				for s in VP2_skels:
					VP2.depthIm = display_skeletons(VP2.depthIm, s, (5000,), skel_type='Kinect')
				VP2.visualize()
				VP2.playback_control()
			VP.visualize()
			# VP.playback_control()



if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-s', '--skel', dest='skel', action="store_true", default=False, help='Enable skeleton')	
	parser.add_option('-d', '--depth', dest='depth', action="store_true", default=False, help='Enable depth images')		
	parser.add_option('-c', '--color', dest='color', action="store_true", default=False, help='Enable color images')	
	parser.add_option('-m', '--mask', dest='mask', action="store_true", default=False, help='Enable enternal mask')
	parser.add_option('-a', '--anonomize', dest='save', action="store_true", default=False, help='Save anonomized RGB image')
	parser.add_option('-f', '--calcFeatures', dest='bgSubtraction', action="store_true", default=False, help='Enable feature extraction')		
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	parser.add_option('-i', '--dev', dest='dev', type='int', default=0, help='Device number')
	(opt, args) = parser.parse_args()

	if opt.viz:
		display_help()

	if len(args) > 0:
		print "Wrong input argument"
	elif opt.depth==False and opt.color==False and opt.skel==False:
		print "You must supply the program with some arguments."
	else:
		main(get_depth=opt.depth, get_skeleton=opt.skel, get_color=opt.color, visualize=opt.viz)

	'''Profiling'''
	# cProfile.runctx('main()', globals(), locals(), filename="ShowSkeletons.profile")
