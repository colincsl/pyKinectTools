
import os
import optparse
import time
import numpy as np
import joblib

from pyKinectTools.utils.SkeletonUtils import msr_to_kinect_skel, get_skel_angles

def smij(skels, confidence=None):
	'''
	Ranks joints based on variance
	Output: ranked joints, ranked variances
	'''

	skel_angles = np.array([get_CAD_skel_angles(x) for x in skels])
	n_joint_angles = len(skel_angles[0])
	skel_angles_variance = np.var(skel_angles, 0)
	skel_angles_variance = np.nan_to_num(skel_angles_variance)

	if confidence is not None:
		idx = np.nonzero(confidence==0)
		skel_angles_variance[idx] *= 0

	joints_ranked = np.argsort(skel_angles_variance)[::-1]
	# skel_angles_variance = [np.var(skel_angles, 0) for i in range(len(skels))]
	# joints_ranked = [np.argsort(x)[::-1] for x in skel_angles_variance]

	# Rank joint angles by max variance
	return joints_ranked, skel_angles_variance
