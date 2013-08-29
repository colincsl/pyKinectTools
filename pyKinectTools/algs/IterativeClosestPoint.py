''' 
Iterative closest point

Colin Lea 
pyKinectTools
2012
'''

import numpy as np
from numpy import dot
from scipy import spatial


def IterativeClosestPoint(pts_new, pts_ref, max_iters=25, min_change=.001, pt_tolerance=5000, return_transform=False):

	# pts_new = y
	# pts_ref = x
	pts_new = pts_new.copy()
	pts_ref = pts_ref.copy()
	
	inital_mean = pts_ref.mean(0)
	# pts_new -= inital_mean
	# pts_ref -= inital_mean

	nn = spatial.cKDTree(pts_ref)
	it = 0
	prevErr = 10**10
	change = np.inf
	R_total = np.eye(3)
	T_total = np.zeros(3)
	R = R_total
	t = T_total
	pts_new_current = np.copy(pts_new)

	err_best = np.inf
	T_best = None
	# T_best = pts_ref - pts_new
	R_best = None


	while it < max_iters and change > min_change:
		# pts_new_current = (dot(R_total, pts_new.T).T + T_total)
		pts_new_current = (dot(R, pts_new_current.T).T + t)
		# scatter(pts_ref[:,0], pts_ref[:,1])
		# scatter(pts_new_current[:,0], pts_new_current[:,1], color='r')
		dists, pts = nn.query(pts_new_current)

		goodPts_new = pts_new_current[dists < pt_tolerance]
		goodPts_ref = pts_ref[pts[dists<pt_tolerance]]

		tmp = 2
		while goodPts_new.shape[0] < 10 and tmp < 10:
			goodPts_new = pts_new_current[dists < pt_tolerance*tmp]
			goodPts_ref = pts_ref[pts[dists<pt_tolerance*tmp]]
			tmp += 1

		R,t = PointcloudRegistration(goodPts_new, goodPts_ref)
		err = np.linalg.norm(goodPts_ref - (dot(R, goodPts_new.T).T + t), 2)

		change = np.abs(prevErr - err)
		prevErr = err
		# print R, R_total
		# print t, T_total
		# print err

		R_total = dot(R, R_total)
		T_total += t

		if err < err_best:
			err_best = err
			R_best = R_total
			T_best = T_total

		it += 1

	# T_best += inital_mean

	if not return_transform:
		return R_best, T_best
	else:
		return R_best, T_best, pts_new_current


def PointcloudRegistration(pts_new, pts_ref):
	''' Uses Arun's SVD-based method
		Input: Nx3 numpy arrays
		Output: R=3x3 rotation matrix, T=3x translation vector 
	'''
	assert pts_new.shape == pts_ref.shape, "Inputs must be of the same dimension"

	''' De-mean '''
	mean_new = pts_new.mean(0)
	mean_ref = pts_ref.mean(0)

	pts_new -= mean_new
	pts_ref -= mean_ref

	H = np.zeros([3,3])
	for i in range(pts_new.shape[0]):
		H += np.outer(pts_new[i], pts_ref[i])

	U,_,Vt = np.linalg.svd(H, full_matrices=0)

	''' Check that it's a rotation '''
	if np.linalg.det(Vt) < 0:
		Vt[2] *= -1

	R = dot(Vt.T, U.T)
	T = mean_ref - dot(R, mean_new)

	return R, T


''' TESTS '''
if 0:

	def RotationZ(ang):
		return np.array([[cos(ang), sin(ang), 0],
						[-sin(ang), cos(ang), 0],
						[0,			0,		 1]])

	eps = .0001
	pts_ref = np.random.random([100,3])*3000

	'''1'''
	T = [100, 0, 0]
	R_in = RotationZ(0)
	pts_new = dot(R_in,pts_ref.T).T + T
	R,t = IterativeClosestPoint(pts_new, pts_ref)
	assert np.all(t+T < eps), "Error too large in test 1"

	'''2'''
	T = [0, 0, 0]
	R_in = RotationZ(10)
	pts_new = dot(R_in,pts_ref.T).T + T
	R,t = IterativeClosestPoint(pts_new, pts_ref)
	assert np.all(dot(R_in,R) - np.eye(3) < eps), "Error too large in test 2"

	'''3'''	
	T = [0, 100, 0]
	R_in = RotationZ(10)
	pts_new = dot(R_in,pts_ref.T).T + T
	R,t = IterativeClosestPoint(pts_new, pts_ref)
	assert np.all(t+T < eps), "Error too large in translation of test 3"
	assert np.all(dot(R_in,R) - np.eye(3) < eps), "Error too large in rotation of test 3"
