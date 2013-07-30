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
from pyKinectTools.dataset_readers.BasePlayer import BasePlayer
from pyKinectTools.utils.SkeletonUtils import *
from pyKinectTools.algs.smij import smij
# from pyKinectTools.dataset_readers.CADPlayer import CADPlayer

""" Debugging """
from IPython import embed


# 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
CAD_DISABLED = [7,9,8,10,13,14]
# -------------------------MAIN------------------------------------------

def get_skels_per_sequence(skels, labels):

	skel_segs = {}
	for i,lab in labels.items():
		start = lab['start']
		stop = lab['stop']
		name = lab['subaction']
		if name in skel_segs.keys():
			skel_segs[name] += [skels[start:stop]]
		else:
			skel_segs[name] = [skels[start:stop]]
	
	return skel_segs


def main():
	# Setup kinect data player

	record = False
	DIR = '/Users/colin/Data/CAD_120/'
	# cam = CADPlayer(base_dir=DIR, get_color=False, get_skeleton=True, subjects=[1], actions=range(10), instances=[1])
	cam = CADPlayer(base_dir=DIR, get_color=False, get_depth=True, get_skeleton=True, 
					subjects=[1,3,4], actions=[1], instances=[0,1,2])

	if record:
		writer = cv2.VideoWriter(filename="CAD_action_{}.avi".format(2), fps=30, 
							frameSize=(640,480),fourcc=cv.CV_FOURCC('I','4','2','0'), isColor=0)	

	framerate = 1
	temporal_window = 1*(14) # 14 FPS
	skel_angles_all = []
	subactivities = []
	skels_subactivity = {}
	while cam.next(framerate):
		# window_start = np.maximum(0, cam.frame-temporal_window)
		
		# Get joint angles
		if cam.frame == 1:
			skels = cam.skel_stack['pos']		
			skel_angles = np.array([get_CAD_skel_angles(x) for x in skels])
			skel_angles = np.nan_to_num(skel_angles)

			tmp = get_skels_per_sequence(skel_angles, cam.subactivity)
			for t in tmp:
				if t not in skels_subactivity:
					skels_subactivity[t] = []
				skels_subactivity[t] += tmp[t]

			skel_angles_all += [skel_angles]
			subactivities += [cam.subactivity]
			cam.next_sequence()
		# break

		# Compute variance over the past second 
		# cam.skels_pos_conf[CAD_DISABLED] = 0
		# smij_features, smij_variance = smij(cam.skel_stack['pos'][window_start:cam.frame], confidence=cam.skels_pos_conf)
		# smij_features = smij_features[:3]
		# print smij_features

		# Add annotations
		cam.depthIm = np.repeat(cam.depthIm[: ,: ,None], 3, -1)
		max_val  = cam.depthIm.max()
		# Label actions
		cv2.putText(cam.depthIm, "Action: "+cam.action, (20,60), cv2.FONT_HERSHEY_DUPLEX, 1, (max_val,0,0), thickness=2)
		cv2.putText(cam.depthIm, "Subaction: "+cam.subaction, (20,85), cv2.FONT_HERSHEY_DUPLEX, 1, (max_val,0,0), thickness=2)
		# Label object affordances
		for o in cam.objects:
			pt1 = tuple(o['topleft'].astype(np.int))
			pt2 = tuple(o['bottomright'].astype(np.int))
			if not all(pt1) or not all(pt2):
				continue
			cv2.rectangle(cam.depthIm, pt1, pt2, (0,0,max_val))
			pt1 = (pt1[0], pt1[1]+15)
			cv2.putText(cam.depthIm, cam.object_names[o["ID"]], pt1, cv2.FONT_HERSHEY_DUPLEX, .6, (0,0,max_val), thickness=1)
		# Label skeleton
		if cam.depth_stack is not None:
			# Hilight active joints
			cam.depthIm = display_skeletons(cam.depthIm, cam.users_uv[0], skel_type='CAD', color=(0,max_val,0), confidence=cam.skels_pos_conf, alt_color=(0,max_val/5,max_val/4))
			
			# Highlight SMIJ joints
			# smij_top = np.ones(cam.users_uv[0].shape[0], np.int)
			# smij_top[smij_features] = 0
			# cam.depthIm = display_skeletons(cam.depthIm, cam.users_uv[0], skel_type='CAD', color=(0,max_val/4,0), confidence=smij_top, alt_color=(0,max_val,0))
		
		cam.visualize(color=True, depth=True)#, depth_bounds=[0,5000])
		
		if record:
			writer.write((cam.depthIm/cam.depthIm.max()*255).astype(np.uint8))

	for it in range(len(skel_angles_all)):
		for i,angs in enumerate(skel_angles_all[it].T):
			figure(i)
			plot(angs)
			title(CAD_JOINTS[i])
	from pylab import *
	import mlpy
	from scipy.interpolate import interp1d
	from scipy.interpolate import *
	from scipy.signal import resample
	# from scipy.signal import cspline1d, cspline1d_eval

	''' it appears to be matching to the wrong subactivity? in DTW '''
	for i_key,key in enumerate(skels_subactivity):
		figure(key)
		# title(key)
		print "{} iterations of {}".format(len(skels_subactivity[key]), key)
		for i_iter in range(len(skels_subactivity[key])):
			print 'iter', i_iter
			for i,ang in enumerate(skels_subactivity[key][i_iter].T):
				x = skels_subactivity[key][0][:,i]
				y = ang
				y = resample(y, len(x))
				error, dtw_mat, y_ind = mlpy.dtw.dtw_std(x, y, dist_only=False)
				error, dtw_mat, y_ind = mlpy.dtw.dtw_subsequence(y, x)
				
				
				subplot(3,4,i+1)
				y_new = y[y_ind[0]]
				x_new = np.linspace(0, 1, len(y_new))
				# poly = polyfit(x_new, y_new, 5)
				# y_spline_ev = poly1d(poly)(x_new)

				nknots = 4
				idx_knots = (np.arange(1,len(x_new)-1,(len(x_new)-2)/np.double(nknots))).astype('int')
				knots = x_new[idx_knots]
				y_spline = splrep(x_new, y_new, t=knots)
				y_spline_ev = splev(np.linspace(0, 1, len(y_new)), y_spline)
				# plot(y_new)
				plot(y_spline_ev)
				# show()
				# plot(y[y_ind[0]])

				# subplot(3,10,i+1 + i_iter*10)
				# plot(x[y_ind[1]])
				# plot(y[y_ind[0]])
				print i,":", len(ang), len(x), len(y[y_ind[0]])
				title(CAD_JOINTS[i])

				if i == 10:
					break
	show()

# Use bezier curves?

import mlpy
x = skel_angles_all[0].T[1]
y = skel_angles_all[1].T[1]
error, dtw_mat, y_ind = mlpy.dtw.dtw_std(y, x, dist_only=False)
plot(y[y_ind[0]-1])
plot(x[y_ind[1]-1])




	if record:
		writer.release()


if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-a', '--anon', dest='anon', action="store_true", default=False, help='Enable anonomization')
	(opt, args) = parser.parse_args()

	main(opt.anon)

