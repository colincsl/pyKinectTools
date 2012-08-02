''' 
Iterative closest point

Colin Lea 
pyKinectTools
2012
'''

import numpy as np
from copy import deepcopy


def IterativeClosestPoint(pointcloud, template, maxIters=100, minChange=.0001):
	# Uses Arun's SVD-based method
	# Output: R=3x3 rotation matrix, T=3x translation vector 

	pointcloudInit = deepcopy(pointcloud)
	R = np.eye(3)
	T = pointcloud.mean(1)

	pointcloud = pointcloud.T - T
	minDists = np.empty(pointcloud.shape[0])
	iters = range(maxIters)
	residual = np.inf
	while iters.pop():

		H = np.asmatrix(np.zeros([3,3]))
		TNew = pointcloud.mean(0)# - template.mean(0)
		T = T + TNew

		pointcloud -= TNew

		for i, val in zip(range(pointcloud.shape[0]),pointcloud):
			dists = np.sum(np.asarray(template - val)**2, 1)
			argMin_ = np.argmin(dists)
			minDists[i] = np.asarray(dists[argMin_])
			H += np.asmatrix(pointcloud[i,:]).T * template[argMin_,:]

		residualNew = np.abs(minDists.sum() / pointcloud.shape[0])
		print "Error: ", residualNew
		if residual - residualNew < minChange:
			break
		residual = residualNew

		U,_,VT = np.linalg.svd(H, full_matrices=0)
		RotNew = (VT.T*U.T)
		R = R*RotNew

		if np.abs(np.linalg.det(RotNew) - 1) > .1:
			print "Error", np.linalg.det(RotNew) - 1

		pointcloud = (R*pointcloudInit).T - T

		# print "Rotated: ", np.arcsin(R[0,1])*180/np.pi

	return R, T
